// runtime/plot/videowriter.cpp — VideoWriter: encode a sequence of captured
// figure frames to a video container via libav (FFmpeg libraries).
//
// MATLAB surface:
//     v = VideoWriter('out.mp4', 'MPEG-4');
//     v.FrameRate = 30;
//     open(v);
//     for k = 1:N
//         plot(...);
//         writeVideo(v, getframe(gcf));
//     end
//     close(v);
//
// The actual H.264 / MJPEG encoding is compiled only when the build sets
// MATLAB_LLVM_WITH_PLOT_FFMPEG=1 (CMake option MATLAB_LLVM_WITH_PLOT_FFMPEG,
// off by default). The C ABI symbols are *always* defined so the JIT can
// resolve matlab_videowriter_* regardless; without libav they return an
// error and print a one-line diagnostic explaining how to enable it.
//
// The container/codec is chosen from the path extension or an explicit
// MATLAB profile name:
//     .mp4 / .m4v  -> "MPEG-4"          -> H.264 in MP4
//     .avi         -> "Motion JPEG AVI" -> MJPEG in AVI

#include "../matlab_plot.h"
#include "frame.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>      // FF_QP2LAMBDA
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}
#endif

namespace {

enum class Profile { Mpeg4H264, MotionJpegAvi };

/* Lowercase ASCII copy of [p, p+n). */
std::string lower(const char *p, int64_t n) {
    std::string s(p, p + (n > 0 ? n : 0));
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

bool ends_with(const std::string &s, std::string_view suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

Profile profile_from_ext(const std::string &lower_path) {
    if (ends_with(lower_path, ".avi")) return Profile::MotionJpegAvi;
    /* .mp4 / .m4v / .mov and anything else default to MPEG-4 H.264. */
    return Profile::Mpeg4H264;
}

Profile profile_from_name(const std::string &lower_name,
                          const std::string &lower_path) {
    if (lower_name.find("motion") != std::string::npos ||
        lower_name.find("jpeg") != std::string::npos ||
        lower_name.find("avi") != std::string::npos)
        return Profile::MotionJpegAvi;
    if (lower_name.find("mpeg-4") != std::string::npos ||
        lower_name.find("mpeg4") != std::string::npos ||
        lower_name.find("mp4") != std::string::npos ||
        lower_name.find("h.264") != std::string::npos ||
        lower_name.find("h264") != std::string::npos)
        return Profile::Mpeg4H264;
    /* Unknown profile name: fall back to the extension default. */
    return profile_from_ext(lower_path);
}

}  // namespace

/* The opaque handle. Defined here (videowriter.cpp owns the struct); the
 * public header only forward-declares matlab_videowriter. */
struct matlab_videowriter {
    std::string path;
    Profile profile = Profile::Mpeg4H264;
    double fps = 30.0;
    double quality = 75.0;   // 0..100, MATLAB convention
    bool opened = false;
    bool started = false;    // encoder initialised from first frame
    bool failed = false;
    int width = 0;
    int height = 0;
    int64_t next_pts = 0;

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
    AVFormatContext *fmt = nullptr;
    const AVCodec *codec = nullptr;
    AVCodecContext *cctx = nullptr;
    AVStream *st = nullptr;
    SwsContext *sws = nullptr;
    AVFrame *yuv = nullptr;
    AVPacket *pkt = nullptr;
#endif
};

namespace {

matlab_videowriter *make_writer(const char *path, int64_t plen, Profile prof) {
    if (!path || plen <= 0) return nullptr;
    auto *v = new matlab_videowriter();
    v->path.assign(path, path + plen);
    v->profile = prof;
    return v;
}

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG

void cleanup(matlab_videowriter *v) {
    if (v->sws) { sws_freeContext(v->sws); v->sws = nullptr; }
    if (v->yuv) { av_frame_free(&v->yuv); }
    if (v->pkt) { av_packet_free(&v->pkt); }
    if (v->cctx) { avcodec_free_context(&v->cctx); }
    if (v->fmt) {
        if (v->fmt->pb && !(v->fmt->oformat->flags & AVFMT_NOFILE))
            avio_closep(&v->fmt->pb);
        avformat_free_context(v->fmt);
        v->fmt = nullptr;
    }
}

/* Drain encoded packets and interleave them into the container. Pass a null
 * frame to flush at end-of-stream. Returns 0 on success. */
int drain(matlab_videowriter *v, AVFrame *frame) {
    if (avcodec_send_frame(v->cctx, frame) < 0) return 1;
    for (;;) {
        int r = avcodec_receive_packet(v->cctx, v->pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) return 1;
        av_packet_rescale_ts(v->pkt, v->cctx->time_base, v->st->time_base);
        v->pkt->stream_index = v->st->index;
        int wr = av_interleaved_write_frame(v->fmt, v->pkt);
        av_packet_unref(v->pkt);
        if (wr < 0) return 1;
    }
    return 0;
}

/* Lazily initialise the encoder once the first frame fixes the dimensions. */
int start_encoder(matlab_videowriter *v, int w, int h) {
    v->width = w;
    v->height = h;

    const bool mjpeg = v->profile == Profile::MotionJpegAvi;
    const char *container = mjpeg ? "avi" : "mp4";
    AVCodecID primary = mjpeg ? AV_CODEC_ID_MJPEG : AV_CODEC_ID_H264;

    if (avformat_alloc_output_context2(&v->fmt, nullptr, container,
                                       v->path.c_str()) < 0 || !v->fmt)
        return 1;

    v->codec = avcodec_find_encoder(primary);
    if (!v->codec && !mjpeg)            // H.264 encoder unavailable -> MPEG-4
        v->codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    if (!v->codec) {
        std::fprintf(stderr,
            "VideoWriter: no encoder available for %s. Rebuild FFmpeg with "
            "the relevant codec (e.g. libx264).\n", container);
        return 1;
    }

    v->st = avformat_new_stream(v->fmt, nullptr);
    if (!v->st) return 1;

    v->cctx = avcodec_alloc_context3(v->codec);
    if (!v->cctx) return 1;
    v->cctx->width = w;
    v->cctx->height = h;
    /* Rational time base from the (possibly fractional) frame rate. */
    AVRational tb = av_d2q(1.0 / (v->fps > 0 ? v->fps : 30.0), 1000000);
    v->cctx->time_base = tb;
    v->cctx->framerate = av_inv_q(tb);
    v->cctx->gop_size = 12;
    v->cctx->pix_fmt = mjpeg ? AV_PIX_FMT_YUVJ420P : AV_PIX_FMT_YUV420P;

    if (mjpeg) {
        /* MJPEG: lower qscale = higher quality (2..31). */
        int qscale = int(std::lround(31.0 - v->quality / 100.0 * 29.0));
        if (qscale < 2) qscale = 2;
        if (qscale > 31) qscale = 31;
        v->cctx->flags |= AV_CODEC_FLAG_QSCALE;
        v->cctx->global_quality = FF_QP2LAMBDA * qscale;
    } else {
        /* H.264: CRF 0 (lossless) .. 51 (worst); map quality 100->0, 0->51. */
        int crf = int(std::lround((100.0 - v->quality) / 100.0 * 51.0));
        if (crf < 0) crf = 0;
        if (crf > 51) crf = 51;
        char crf_s[8];
        std::snprintf(crf_s, sizeof crf_s, "%d", crf);
        av_opt_set(v->cctx->priv_data, "crf", crf_s, 0);
        av_opt_set(v->cctx->priv_data, "preset", "medium", 0);
    }

    if (v->fmt->oformat->flags & AVFMT_GLOBALHEADER)
        v->cctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(v->cctx, v->codec, nullptr) < 0) return 1;

    if (avcodec_parameters_from_context(v->st->codecpar, v->cctx) < 0) return 1;
    v->st->time_base = v->cctx->time_base;

    if (!(v->fmt->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&v->fmt->pb, v->path.c_str(), AVIO_FLAG_WRITE) < 0)
            return 1;
    }
    if (avformat_write_header(v->fmt, nullptr) < 0) return 1;

    v->yuv = av_frame_alloc();
    if (!v->yuv) return 1;
    v->yuv->format = v->cctx->pix_fmt;
    v->yuv->width = w;
    v->yuv->height = h;
    if (av_frame_get_buffer(v->yuv, 0) < 0) return 1;

    v->pkt = av_packet_alloc();
    if (!v->pkt) return 1;

    /* Cairo gives premultiplied ARGB32 == BGRA bytes on little-endian. */
    v->sws = sws_getContext(w, h, AV_PIX_FMT_BGRA, w, h, v->cctx->pix_fmt,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!v->sws) return 1;

    v->started = true;
    return 0;
}

#endif  // MATLAB_LLVM_WITH_PLOT_FFMPEG

void warn_disabled_once() {
    static bool warned = false;
    if (warned) return;
    warned = true;
    std::fprintf(stderr,
        "VideoWriter: this matlabc was built without video support. "
        "Rebuild with -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON (requires the FFmpeg "
        "dev libraries: libavcodec/libavformat/libavutil/libswscale).\n");
}

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
/* Same env gate the figure-emit path uses (figure.cpp / c_api.cpp). Kept
 * local instead of shared so videowriter.cpp doesn't reach into c_api's
 * anonymous namespace — it's a one-line getenv either way. */
bool ide_video_enabled() {
    const char *v = std::getenv("MATLAB_LLVM_IDE_FIGURES");
    return v && v[0] == '1';
}

/* Announce a finished video file to the IDE, mirroring the figure-emit
 * protocol but for an on-disk artifact rather than an inline bitmap:
 *
 *   ___MF_VID___ w=<px> h=<px> fps=<rate> path=<absolute path>
 *
 * `path=` is emitted LAST and runs to end-of-line so paths containing
 * spaces survive intact (the IDE parser splits the w/h/fps tokens, then
 * takes everything after `path=` verbatim). Only called once the trailer
 * has been written, so the file is complete and playable when this lands.
 * No-op unless the IDE env gate is set. */
void emit_ide_video(const matlab_videowriter *v) {
    if (!ide_video_enabled()) return;
    char *resolved = ::realpath(v->path.c_str(), nullptr);
    const char *p = resolved ? resolved : v->path.c_str();
    std::fprintf(stdout, "___MF_VID___ w=%d h=%d fps=%g path=%s\n",
                 v->width, v->height, v->fps, p);
    std::fflush(stdout);
    if (resolved) std::free(resolved);
}
#endif  // MATLAB_LLVM_WITH_PLOT_FFMPEG

}  // namespace

extern "C" {

matlab_videowriter *matlab_videowriter_new(const char *path, int64_t plen) {
    if (!path || plen <= 0) return nullptr;
    return make_writer(path, plen, profile_from_ext(lower(path, plen)));
}

matlab_videowriter *matlab_videowriter_new_profile(const char *path,
                                                   int64_t plen,
                                                   const char *profile,
                                                   int64_t prlen) {
    if (!path || plen <= 0) return nullptr;
    Profile prof =
        (profile && prlen > 0)
            ? profile_from_name(lower(profile, prlen), lower(path, plen))
            : profile_from_ext(lower(path, plen));
    return make_writer(path, plen, prof);
}

void matlab_videowriter_set_framerate(matlab_videowriter *v, double fps) {
    if (v && !v->started && fps > 0) v->fps = fps;
}

void matlab_videowriter_set_quality(matlab_videowriter *v, double q) {
    if (v && !v->started) {
        if (q < 0) q = 0;
        if (q > 100) q = 100;
        v->quality = q;
    }
}

int matlab_videowriter_open(matlab_videowriter *v) {
    if (!v) return 1;
    v->opened = true;
    return 0;
}

int matlab_videowriter_write(matlab_videowriter *v, matlab_frame *frame) {
    if (!v || !frame || frame->pixels.empty()) return 1;
    if (!v->opened) {
        std::fprintf(stderr, "VideoWriter: writeVideo before open().\n");
        return 1;
    }
    if (v->failed) return 1;

#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
    if (!v->started) {
        if (start_encoder(v, frame->width, frame->height) != 0) {
            v->failed = true;
            std::fprintf(stderr, "VideoWriter: failed to start encoder for %s.\n",
                         v->path.c_str());
            return 1;
        }
    }
    if (frame->width != v->width || frame->height != v->height) {
        std::fprintf(stderr,
            "VideoWriter: frame size %dx%d does not match video %dx%d "
            "(keep the figure size fixed across frames).\n",
            frame->width, frame->height, v->width, v->height);
        return 1;
    }
    if (av_frame_make_writable(v->yuv) < 0) return 1;
    const uint8_t *src[1] = { frame->pixels.data() };
    int src_stride[1] = { frame->stride };
    sws_scale(v->sws, src, src_stride, 0, v->height, v->yuv->data,
              v->yuv->linesize);
    v->yuv->pts = v->next_pts++;
    if (drain(v, v->yuv) != 0) {
        v->failed = true;
        return 1;
    }
    return 0;
#else
    (void)frame;
    warn_disabled_once();
    v->failed = true;
    return 1;
#endif
}

int matlab_videowriter_close(matlab_videowriter *v) {
    if (!v) return 1;
    int rc = 0;
#ifdef MATLAB_LLVM_WITH_PLOT_FFMPEG
    if (v->started && !v->failed) {
        if (drain(v, nullptr) != 0) rc = 1;           // flush encoder
        if (av_write_trailer(v->fmt) < 0) rc = 1;
        if (rc == 0) emit_ide_video(v);               // surface to the IDE
    } else if (v->started) {
        rc = 1;
    }
    cleanup(v);
#else
    /* Opened but built without libav: no file was produced — report it. */
    if (v->opened) {
        warn_disabled_once();
        rc = 1;
    }
#endif
    delete v;
    return rc;
}

}  // extern "C"
