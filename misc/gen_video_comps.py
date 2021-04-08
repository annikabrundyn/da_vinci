import os
import random
import argparse

from moviepy.editor import VideoFileClip, ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips


def make_comp_clip(q_num, a, b, t_start, clip_len, input_dir):

    pad_top = pad_bottom = (720 - 192) // 2
    pad_left = pad_right = (1280 - 768) // 2

    t_end = t_start + clip_len

    clip_a = VideoFileClip(os.path.join(input_dir, f"{a}_last.mp4")).subclip(t_start, t_end)
    clip_b = VideoFileClip(os.path.join(input_dir, f"{b}_last.mp4")).subclip(t_start, t_end)

    clip_a = clip_a.margin(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)
    clip_b = clip_b.margin(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)

    black = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=1)

    txt_clip_a = TextClip("Video A", fontsize=35, font='Liberation-Sans-Narrow', color='white')
    txt_clip_a = txt_clip_a.set_position(("center", "bottom")).set_duration(5)

    txt_clip_b = TextClip("Video B", fontsize=35, font='Liberation-Sans-Narrow', color='white')
    txt_clip_b = txt_clip_b.set_position(("center", "bottom")).set_duration(5)

    clip_a = CompositeVideoClip([clip_a, txt_clip_a])
    clip_b = CompositeVideoClip([clip_b, txt_clip_b])

    final_clip = concatenate_videoclips([black, clip_a, black, clip_b])

    txt_clip_q = TextClip(f"Q{q_num}", fontsize=40, font='Liberation-Sans', color='white')
    txt_clip_q = txt_clip_q.set_position(("left", "top")).set_duration(18)
    final_clip = CompositeVideoClip([final_clip, txt_clip_q])
    return final_clip


def make_all_comp_clips(inputs, input_dir, out_dir):

    for (i, a, b, t_start, clip_len) in inputs:
        print(i)
        print(a)
        print(b)

        # make
        output_clip = make_comp_clip(i, a, b, t_start, clip_len, input_dir)

        # save
        output_clip.write_videofile(os.path.join(out_dir, f"Q{i}.mp4"))
