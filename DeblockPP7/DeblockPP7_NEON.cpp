#include "DeblockPP7.hpp"

#ifdef VS_TARGET_CPU_ARM_NEON
#include <arm_neon.h>

static inline void dctA(const int * srcp, int * dstp, const int stride) noexcept {
    int32x4_t s0 = vaddq_s32(vld1q_s32(srcp + 0 * stride), vld1q_s32(srcp + 6 * stride));
    int32x4_t s1 = vaddq_s32(vld1q_s32(srcp + 1 * stride), vld1q_s32(srcp + 5 * stride));
    int32x4_t s2 = vaddq_s32(vld1q_s32(srcp + 2 * stride), vld1q_s32(srcp + 4 * stride));
    int32x4_t s3 = vld1q_s32(srcp + 3 * stride);
    int32x4_t s = vaddq_s32(s3, s3);
    s3 = vsubq_s32(s, s0);
    s0 = vaddq_s32(s, s0);
    s = vaddq_s32(s2, s1);
    s2 = vsubq_s32(s2, s1);

    int32x4x4_t r;
    r.val[0] = vaddq_s32(s0, s);
    r.val[1] = vaddq_s32(vaddq_s32(s3, s3), s2);
    r.val[2] = vsubq_s32(s0, s);
    r.val[3] = vsubq_s32(s3, vaddq_s32(s2, s2));
    vst4q_s32(dstp, r);
}

static inline void dctA(const float * srcp, float * dstp, const int stride) noexcept {
    float32x4_t s0 = vmulq_n_f32(vaddq_f32(vld1q_f32(srcp + 0 * stride), vld1q_f32(srcp + 6 * stride)), 255.f);
    float32x4_t s1 = vmulq_n_f32(vaddq_f32(vld1q_f32(srcp + 1 * stride), vld1q_f32(srcp + 5 * stride)), 255.f);
    float32x4_t s2 = vmulq_n_f32(vaddq_f32(vld1q_f32(srcp + 2 * stride), vld1q_f32(srcp + 4 * stride)), 255.f);
    float32x4_t s3 = vmulq_n_f32(vld1q_f32(srcp + 3 * stride), 255.f);
    float32x4_t s = vaddq_f32(s3, s3);
    s3 = vsubq_f32(s, s0);
    s0 = vaddq_f32(s, s0);
    s = vaddq_f32(s2, s1);
    s2 = vsubq_f32(s2, s1);

    float32x4x4_t r;
    r.val[0] = vaddq_f32(s0, s);
    r.val[1] = vaddq_f32(vaddq_f32(s3, s3), s2);
    r.val[2] = vsubq_f32(s0, s);
    r.val[3] = vsubq_f32(s3, vaddq_f32(s2, s2));
    vst4q_f32(dstp, r);
}

static inline void dctB(const int * srcp, int32x4_t block[4]) noexcept {
    int32x4_t s0 = vaddq_s32(vld1q_s32(srcp + 0 * 4), vld1q_s32(srcp + 6 * 4));
    int32x4_t s1 = vaddq_s32(vld1q_s32(srcp + 1 * 4), vld1q_s32(srcp + 5 * 4));
    int32x4_t s2 = vaddq_s32(vld1q_s32(srcp + 2 * 4), vld1q_s32(srcp + 4 * 4));
    int32x4_t s3 = vld1q_s32(srcp + 3 * 4);
    int32x4_t s = vaddq_s32(s3, s3);
    s3 = vsubq_s32(s, s0);
    s0 = vaddq_s32(s, s0);
    s = vaddq_s32(s2, s1);
    s2 = vsubq_s32(s2, s1);
    block[0] = vaddq_s32(s0, s);
    block[1] = vaddq_s32(vaddq_s32(s3, s3), s2);
    block[2] = vsubq_s32(s0, s);
    block[3] = vsubq_s32(s3, vaddq_s32(s2, s2));
}

static inline void dctB(const float * srcp, float32x4_t block[4]) noexcept {
    float32x4_t s0 = vaddq_f32(vld1q_f32(srcp + 0 * 4), vld1q_f32(srcp + 6 * 4));
    float32x4_t s1 = vaddq_f32(vld1q_f32(srcp + 1 * 4), vld1q_f32(srcp + 5 * 4));
    float32x4_t s2 = vaddq_f32(vld1q_f32(srcp + 2 * 4), vld1q_f32(srcp + 4 * 4));
    float32x4_t s3 = vld1q_f32(srcp + 3 * 4);
    float32x4_t s = vaddq_f32(s3, s3);
    s3 = vsubq_f32(s, s0);
    s0 = vaddq_f32(s, s0);
    s = vaddq_f32(s2, s1);
    s2 = vsubq_f32(s2, s1);
    block[0] = vaddq_f32(s0, s);
    block[1] = vaddq_f32(vaddq_f32(s3, s3), s2);
    block[2] = vsubq_f32(s0, s);
    block[3] = vsubq_f32(s3, vaddq_f32(s2, s2));
}

// Thresholds every coefficient of block row r (block[r * 4 + 0..3]) at once.
// Lane 0 of row 0 is the DC coefficient, which is always kept unmodified.
template<int mode>
static inline int32x4_t thresholdInt(const int32x4_t b, const uint32x4_t t1, const uint32x4_t t2) noexcept {
    const uint32x4_t ub = vreinterpretq_u32_s32(b);
    const uint32x4_t pass = vcgtq_u32(vaddq_u32(ub, t1), t2);

    if (mode == 0)
        return vandq_s32(b, vreinterpretq_s32_u32(pass));

    const int32x4_t st1 = vreinterpretq_s32_u32(t1);
    int32x4_t soft = vbslq_s32(vcgtq_s32(b, vdupq_n_s32(0)), vsubq_s32(b, st1), vaddq_s32(b, st1));

    if (mode == 2) {
        const uint32x4_t hard = vcgtq_u32(vaddq_u32(ub, t2), vaddq_u32(t2, t2));
        soft = vbslq_s32(hard, b, vaddq_s32(soft, soft));
    }

    return vandq_s32(soft, vreinterpretq_s32_u32(pass));
}

template<int mode>
static inline float32x4_t thresholdFloat(const float32x4_t b, const uint32x4_t t1, const uint32x4_t t2, const float32x4_t ft1) noexcept {
    const uint32x4_t ub = vreinterpretq_u32_s32(vcvtq_s32_f32(b));
    const uint32x4_t pass = vcgtq_u32(vaddq_u32(ub, t1), t2);

    float32x4_t r;
    if (mode == 0) {
        r = b;
    } else {
        r = vbslq_f32(vcgtq_f32(b, vdupq_n_f32(0.f)), vsubq_f32(b, ft1), vaddq_f32(b, ft1));

        if (mode == 2) {
            const uint32x4_t hard = vcgtq_u32(vaddq_u32(ub, t2), vaddq_u32(t2, t2));
            r = vbslq_f32(hard, b, vaddq_f32(r, r));
        }
    }

    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(r), pass));
}

template<typename T, int mode>
static void filterPlane(const T * srcp, T * VS_RESTRICT dstp, int * buffer, const int width, const int height, const int srcStride, const int stride,
                        const DeblockPP7Data * const VS_RESTRICT d) noexcept {
    int * VS_RESTRICT p_src = buffer + stride * 8;
    int * VS_RESTRICT temp = buffer + 16;

    for (int y = 0; y < height; y++) {
        const int index = stride * (8 + y) + 8;
        std::copy_n(srcp + srcStride * y, width, p_src + index);
        for (int x = 0; x < 8; x++) {
            p_src[index - 1 - x] = p_src[index + x];
            p_src[index + width + x] = p_src[index + width - 1 - x];
        }
    }
    for (int y = 0; y < 8; y++) {
        memcpy(p_src + stride * (7 - y), p_src + stride * (8 + y), stride * sizeof(int));
        memcpy(p_src + stride * (height + 8 + y), p_src + stride * (height + 7 - y), stride * sizeof(int));
    }

    const uint32x4_t dcMask = vsetq_lane_u32(UINT32_MAX, vdupq_n_u32(0), 0);
    uint32x4_t t1[4], t2[4];
    int32x4_t factor[4];
    for (int i = 0; i < 4; i++) {
        t1[i] = vld1q_u32(d->thresh + i * 4);
        t2[i] = vaddq_u32(t1[i], t1[i]);
        factor[i] = vmovl_s16(vld1_s16(d->factor + i * 4));
    }

    for (int y = 0; y < height; y++) {
        for (int x = -8; x < 0; x += 4) {
            const int index = (stride + 1) * (8 - 3) + stride * y + 8 + x;
            int * VS_RESTRICT tp = temp + 4 * x;

            dctA(p_src + index, tp + 4 * 8, stride);
        }

        for (int x = 0; x < width; x++) {
            const int index = (stride + 1) * (8 - 3) + stride * y + 8 + x;
            int * VS_RESTRICT tp = temp + 4 * x;

            if (!(x & 3))
                dctA(p_src + index, tp + 4 * 8, stride);

            int32x4_t block[4];
            dctB(tp, block);

            int32x4_t c = vbslq_s32(dcMask, block[0], thresholdInt<mode>(block[0], t1[0], t2[0]));
            int64x2_t acc = vmull_s32(vget_low_s32(c), vget_low_s32(factor[0]));
            acc = vmlal_high_s32(acc, c, factor[0]);
            for (int i = 1; i < 4; i++) {
                c = thresholdInt<mode>(block[i], t1[i], t2[i]);
                acc = vmlal_s32(acc, vget_low_s32(c), vget_low_s32(factor[i]));
                acc = vmlal_high_s32(acc, c, factor[i]);
            }

            int64_t v = vaddvq_s64(acc);
            v = (v + (1 << 17)) >> 18;
            if (static_cast<unsigned>(v) > d->peak)
                v = (v < 0) ? 0 : d->peak;

            dstp[srcStride * y + x] = static_cast<T>(v);
        }
    }
}

template<int mode>
static void filterPlane(const float * srcp, float * VS_RESTRICT dstp, float * buffer, const int width, const int height, const int srcStride, const int stride,
                        const DeblockPP7Data * const VS_RESTRICT d) noexcept {
    float * VS_RESTRICT p_src = buffer + stride * 8;
    float * VS_RESTRICT temp = buffer + 16;

    for (int y = 0; y < height; y++) {
        const int index = stride * (8 + y) + 8;
        std::copy_n(srcp + srcStride * y, width, p_src + index);
        for (int x = 0; x < 8; x++) {
            p_src[index - 1 - x] = p_src[index + x];
            p_src[index + width + x] = p_src[index + width - 1 - x];
        }
    }
    for (int y = 0; y < 8; y++) {
        memcpy(p_src + stride * (7 - y), p_src + stride * (8 + y), stride * sizeof(float));
        memcpy(p_src + stride * (height + 8 + y), p_src + stride * (height + 7 - y), stride * sizeof(float));
    }

    const uint32x4_t dcMask = vsetq_lane_u32(UINT32_MAX, vdupq_n_u32(0), 0);
    uint32x4_t t1[4], t2[4];
    float32x4_t ft1[4], factor[4];
    for (int i = 0; i < 4; i++) {
        t1[i] = vld1q_u32(d->thresh + i * 4);
        t2[i] = vaddq_u32(t1[i], t1[i]);
        ft1[i] = vcvtq_f32_u32(t1[i]);
        factor[i] = vcvtq_f32_s32(vmovl_s16(vld1_s16(d->factor + i * 4)));
    }

    for (int y = 0; y < height; y++) {
        for (int x = -8; x < 0; x += 4) {
            const int index = (stride + 1) * (8 - 3) + stride * y + 8 + x;
            float * VS_RESTRICT tp = temp + 4 * x;

            dctA(p_src + index, tp + 4 * 8, stride);
        }

        for (int x = 0; x < width; x++) {
            const int index = (stride + 1) * (8 - 3) + stride * y + 8 + x;
            float * VS_RESTRICT tp = temp + 4 * x;

            if (!(x & 3))
                dctA(p_src + index, tp + 4 * 8, stride);

            float32x4_t block[4];
            dctB(tp, block);

            float32x4_t c = vbslq_f32(dcMask, block[0], thresholdFloat<mode>(block[0], t1[0], t2[0], ft1[0]));
            float32x4_t acc = vmulq_f32(c, factor[0]);
            for (int i = 1; i < 4; i++) {
                c = thresholdFloat<mode>(block[i], t1[i], t2[i], ft1[i]);
                acc = vmlaq_f32(acc, c, factor[i]);
            }

            dstp[srcStride * y + x] = vaddvq_f32(acc) * ((1.f / (1 << 18)) * (1.f / 255.f));
        }
    }
}

template<typename T>
void pp7Filter_neon(const VSFrame * src, VSFrame * dst, const DeblockPP7Data * const VS_RESTRICT d, int * VS_RESTRICT buffer, const VSAPI * vsapi) noexcept {

    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int srcStride = static_cast<int>(vsapi->getStride(src, plane) / sizeof(T));
            const int stride = d->stride[plane];
            const T * srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
            T * VS_RESTRICT dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

            if (d->mode == 0)
                filterPlane<T, 0>(srcp, dstp, buffer, width, height, srcStride, stride, d);
            else if (d->mode == 1)
                filterPlane<T, 1>(srcp, dstp, buffer, width, height, srcStride, stride, d);
            else
                filterPlane<T, 2>(srcp, dstp, buffer, width, height, srcStride, stride, d);
        }
    }
}

template void pp7Filter_neon<uint8_t>(const VSFrame *, VSFrame *, const DeblockPP7Data * const VS_RESTRICT, int * VS_RESTRICT, const VSAPI *) noexcept;
template void pp7Filter_neon<uint16_t>(const VSFrame *, VSFrame *, const DeblockPP7Data * const VS_RESTRICT, int * VS_RESTRICT, const VSAPI *) noexcept;

template<>
void pp7Filter_neon<float>(const VSFrame * src, VSFrame * dst, const DeblockPP7Data * const VS_RESTRICT d, int * VS_RESTRICT intBuffer, const VSAPI * vsapi) noexcept {
    float * buffer = reinterpret_cast<float *>(intBuffer);

    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int srcStride = static_cast<int>(vsapi->getStride(src, plane) / sizeof(float));
            const int stride = d->stride[plane];
            const float * srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane));
            float * VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

            if (d->mode == 0)
                filterPlane<0>(srcp, dstp, buffer, width, height, srcStride, stride, d);
            else if (d->mode == 1)
                filterPlane<1>(srcp, dstp, buffer, width, height, srcStride, stride, d);
            else
                filterPlane<2>(srcp, dstp, buffer, width, height, srcStride, stride, d);
        }
    }
}
#endif
