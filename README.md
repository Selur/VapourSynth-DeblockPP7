Description
===========

Variant of the spp filter in MPlayer, similar to spp=6 with 7 point DCT where only the center sample is used after IDCT.

This version uses the VapourSynth API 4 (VapourSynth R55 or newer).


Usage
=====

    pp7.DeblockPP7(clip clip[, float qp=2.0, int mode=0, int opt=0, int[] planes])

* clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

* qp: Constant quantization parameter. It accepts a value in range 1.0 to 63.0.

* mode:
  * 0 = hard thresholding
  * 1 = soft thresholding (better deringing, but blurrier)
  * 2 = medium thresholding (compromise between hard and soft)

* opt: Sets which cpu optimizations to use.
  * 0 = auto detect
  * 1 = use c
  * 2 = use sse2
  * 3 = use sse4.1

  The SSE code paths only exist in x86/x86_64 builds; other builds (e.g. macOS arm64) always use the C code.

* planes: A list of the planes to process. By default all planes are processed.


Installation
============

Every push builds Python wheels for Windows x64, Linux x86_64 and macOS arm64 (see `.github/workflows/build-wheels.yml`); tagged releases (`v*`) attach them to a GitHub release. The wheel installs the plugin into the VapourSynth plugin folder of the Python package, so it is autoloaded:

```
pip install vapoursynth_deblockpp7-*.whl
```


Compilation
===========

Meson and Ninja are required. The VapourSynth API 4 headers are bundled in `include/vapoursynth`, a system installation of VapourSynth is optional and preferred when found.

```
meson setup build
ninja -C build
```

On macOS the plugin is built as `libdeblockpp7.dylib`, which is the only extension VapourSynth autoloads there.


Testing
=======

`test/test_deblockpp7.py` runs the plugin on synthetic clips: it checks all supported formats and modes, that the SSE2/SSE4.1 code paths match the C code, the `planes` parameter, padding of odd dimensions and parameter validation. It needs the `vapoursynth` Python module and `numpy`:

```
python3 test/test_deblockpp7.py build/libdeblockpp7.so
```
