#!/usr/bin/env python3
"""Functional tests for the DeblockPP7 VapourSynth plugin.

Usage: python3 test/test_deblockpp7.py [path/to/libdeblockpp7.so]

Without arguments the plugin is expected to be autoloaded (e.g. from the
installed wheel). Requires the vapoursynth Python module and numpy.
"""
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
if len(sys.argv) > 1:
    core.std.LoadPlugin(sys.argv[1])

pp7 = core.pp7.DeblockPP7

WIDTH, HEIGHT, FRAMES = 96, 64, 3
SEED = 1234


def blocky_clip(fmt, width=WIDTH, height=HEIGHT):
    """A deterministic 'blocky' picture: a smooth gradient with a small random
    offset per 8x8 block (blocking artifacts) plus mild noise."""
    base = core.std.BlankClip(width=width, height=height, length=FRAMES, format=fmt)
    f = base.format
    rng = np.random.default_rng(SEED)
    planes = []
    for p in range(f.num_planes):
        w = width >> (f.subsampling_w if p else 0)
        h = height >> (f.subsampling_h if p else 0)
        frames = []
        for n in range(FRAMES):
            yy, xx = np.mgrid[0:h, 0:w]
            gradient = 40 + 170 * (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2
            blocks = rng.integers(-8, 9, size=((h + 7) // 8, (w + 7) // 8))
            img = gradient + np.kron(blocks, np.ones((8, 8)))[:h, :w] + rng.normal(0, 1.5, size=(h, w)) + n
            img = np.clip(np.rint(img), 0, 255)
            if f.sample_type == vs.FLOAT:
                frames.append((img / 255.0).astype(np.float32))
            else:
                frames.append((img.astype(np.int64) << (f.bits_per_sample - 8)).astype(
                    np.uint8 if f.bits_per_sample == 8 else np.uint16))
        planes.append(frames)

    def fill(n, f):
        f = f.copy()
        for p in range(f.format.num_planes):
            np.asarray(f[p])[:] = planes[p][n]
        return f

    return core.std.ModifyFrame(base, base, fill)


def to_arrays(clip):
    out = []
    for n in range(clip.num_frames):
        f = clip.get_frame(n)
        out.append([np.array(f[p], dtype=np.float64) for p in range(f.format.num_planes)])
    return out


def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)
    print("ok  ", msg)


def expect_error(fn, msg):
    try:
        fn()
    except vs.Error:
        check(True, msg)
    else:
        raise SystemExit("FAIL: no error raised: " + msg)


def max_diff(a, b):
    return max(float(np.abs(x - y).max()) for fa, fb in zip(a, b) for x, y in zip(fa, fb))


def main():
    formats = [vs.YUV420P8, vs.YUV444P10, vs.YUV422P16, vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.RGBS]

    # 1. Every format, every mode: the output keeps the clip's properties, the
    #    picture is changed and stays in range.
    for fmt in formats:
        src = blocky_clip(fmt)
        inp = to_arrays(src)
        f = core.get_video_format(fmt)
        peak = 1.0 if f.sample_type == vs.FLOAT else (1 << f.bits_per_sample) - 1
        for mode in range(3):
            out = pp7(src, qp=8, mode=mode)
            check(out.format == src.format and out.width == WIDTH and out.height == HEIGHT
                  and out.num_frames == FRAMES, "%s mode=%d keeps format and size" % (f.name, mode))
            arr = to_arrays(out)
            check(max_diff(arr, inp) > 0, "%s mode=%d changes the picture" % (f.name, mode))
            check(all(a.min() >= 0 and a.max() <= peak for fa in arr for a in fa),
                  "%s mode=%d output stays within range" % (f.name, mode))

    # 2. The filter smooths block edges: the energy of the horizontal and
    #    vertical differences at 8x8 block borders drops.
    src = blocky_clip(vs.GRAY8)
    inp = to_arrays(src)[0][0]
    out = to_arrays(pp7(src, qp=12))[0][0]

    def border_energy(a):
        return float(np.abs(np.diff(a, axis=1))[:, 7::8].mean() + np.abs(np.diff(a, axis=0))[7::8, :].mean())

    check(border_energy(out) < 0.5 * border_energy(inp),
          "block borders are smoothed (%.2f -> %.2f)" % (border_energy(inp), border_energy(out)))

    # 3. The SIMD code paths must match the C code path (exactly for integer
    #    samples, closely for float samples). On non x86 builds opt=2/3 fall
    #    back to the C code, so the comparison is trivially true there.
    for fmt in [vs.YUV420P8, vs.YUV444P10, vs.GRAY16, vs.GRAYS]:
        src = blocky_clip(fmt)
        f = core.get_video_format(fmt)
        tol = 1e-5 if f.sample_type == vs.FLOAT else 0
        for mode in range(3):
            ref = to_arrays(pp7(src, qp=6, mode=mode, opt=1))
            for opt in (2, 3):
                diff = max_diff(to_arrays(pp7(src, qp=6, mode=mode, opt=opt)), ref)
                check(diff <= tol, "%s mode=%d opt=%d matches the C code (max diff %g)" % (f.name, mode, opt, diff))
        diff = max_diff(to_arrays(pp7(src, qp=6, opt=0)), to_arrays(pp7(src, qp=6, opt=1)))
        check(diff <= tol, "%s opt=0 matches the C code (max diff %g)" % (f.name, diff))

    # 4. planes: unlisted planes are copied unchanged.
    src = blocky_clip(vs.YUV444P8)
    inp = to_arrays(src)
    out = to_arrays(pp7(src, qp=8, planes=[0]))
    check(max_diff([[a[0]] for a in out], [[a[0]] for a in inp]) > 0, "planes=[0] filters the luma plane")
    check(all(np.array_equal(o[p], i[p]) for o, i in zip(out, inp) for p in (1, 2)),
          "planes=[0] leaves the chroma planes untouched")

    # 5. Dimensions that are not a multiple of 16 are padded internally and
    #    cropped back: the output keeps the original size and is filtered.
    src = blocky_clip(vs.GRAY8, width=50, height=35)
    out = pp7(src, qp=8)
    check(out.width == 50 and out.height == 35, "odd dimensions are preserved")
    arr = to_arrays(out)
    check(max_diff(arr, to_arrays(src)) > 0 and all(a.max() <= 255 for fa in arr for a in fa),
          "odd dimensions are filtered")

    # 6. A flat clip stays flat.
    for fmt in [vs.GRAY8, vs.YUV420P10, vs.GRAYS]:
        f = core.get_video_format(fmt)
        value = 0.5 if f.sample_type == vs.FLOAT else 1 << (f.bits_per_sample - 1)
        clip = core.std.BlankClip(width=WIDTH, height=HEIGHT, length=2, format=fmt, color=[value] * f.num_planes)
        arr = to_arrays(pp7(clip, qp=8))
        check(all(np.allclose(a, value, atol=1e-6) for fa in arr for a in fa), "%s flat clip is unchanged" % f.name)

    # 7. Parameter validation.
    src = blocky_clip(vs.YUV420P8)
    expect_error(lambda: pp7(src, qp=0.5), "qp below 1.0 is rejected")
    expect_error(lambda: pp7(src, qp=64), "qp above 63.0 is rejected")
    expect_error(lambda: pp7(src, mode=3), "mode=3 is rejected")
    expect_error(lambda: pp7(src, opt=4), "opt=4 is rejected")
    expect_error(lambda: pp7(src, planes=[3]), "plane index out of range is rejected")
    expect_error(lambda: pp7(src, planes=[0, 0]), "duplicate plane is rejected")
    expect_error(lambda: pp7(core.std.BlankClip(format=vs.GRAYH)), "16 bit float input is rejected")

    print("all tests passed")


if __name__ == "__main__":
    main()
