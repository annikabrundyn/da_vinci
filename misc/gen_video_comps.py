import os
import random
import argparse
from tqdm import tqdm

from moviepy.editor import VideoFileClip, ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips
import moviepy.video.fx.all as vfx


def make_snippet(q_num, a, b, t_start, clip_len, input_dir, speed_factor):
    total_len = clip_len + 3
    t_end = t_start + clip_len

    black = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=total_len)

    txt_clip_q = TextClip(f"Q{q_num}", fontsize=40, font='Liberation-Sans', color='white')

    clip_a = VideoFileClip(os.path.join(input_dir, f"{a}_last.mp4")).subclip(t_start, t_end)
    clip_b = VideoFileClip(os.path.join(input_dir, f"{b}_last.mp4")).subclip(t_start, t_end)

    txt_clip_a = TextClip("Video A:", fontsize=30, font='Liberation-Sans-Narrow', color='white')
    txt_clip_b = TextClip("Video B:", fontsize=30, font='Liberation-Sans-Narrow', color='white')

    video = CompositeVideoClip([black,
                                txt_clip_q.set_position(("left", "top")).set_duration(total_len),
                                clip_a.set_start(1).set_position((300, 120)),
                                txt_clip_a.set_start(1).set_position((120, 130)).set_duration(clip_len),
                                clip_b.set_start(1).set_position((300, 400)),
                                txt_clip_b.set_start(1).set_position((120, 410)).set_duration(clip_len)])

    video.fx(vfx.speedx, speed_factor)

    return video


def make_all_comp_clips(inputs, input_dir, out_dir, speed_factor=0.5):

    for (i, a, b, t_start, clip_len) in tqdm(inputs):
        # make
        output_clip = make_snippet(i, a, b, t_start, clip_len, input_dir, speed_factor)

        # save
        output_clip.write_videofile(os.path.join(out_dir, f"Q{i}.mp4"), verbose=False)
