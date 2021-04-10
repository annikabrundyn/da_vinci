import os
import random
import argparse
from tqdm import tqdm

from moviepy.editor import VideoFileClip, ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips


def make_snippet(q_num, a, b, t_start, clip_len, input_dir):
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

    video.fx(vfx.speedx, 0.75)

    return video

# def make_comp_clip(q_num, a, b, t_start, clip_len, input_dir):
#
#     pad_top = pad_bottom = (720 - 192) // 2
#     pad_left = pad_right = (1280 - 768) // 2
#
#     t_end = t_start + clip_len
#
#     clip_a = VideoFileClip(os.path.join(input_dir, f"{a}_last.mp4")).subclip(t_start, t_end)
#     clip_b = VideoFileClip(os.path.join(input_dir, f"{b}_last.mp4")).subclip(t_start, t_end)
#
#     clip_a = clip_a.margin(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)
#     clip_b = clip_b.margin(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)
#
#     black = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=1)
#
#     txt_clip_a = TextClip("Video A", fontsize=35, font='Liberation-Sans-Narrow', color='white')
#     txt_clip_a = txt_clip_a.set_position(("center", "bottom")).set_duration(clip_len)
#
#     txt_clip_b = TextClip("Video B", fontsize=35, font='Liberation-Sans-Narrow', color='white')
#     txt_clip_b = txt_clip_b.set_position(("center", "bottom")).set_duration(clip_len)
#
#     clip_a = CompositeVideoClip([clip_a, txt_clip_a])
#     clip_b = CompositeVideoClip([clip_b, txt_clip_b])
#
#     final_clip = concatenate_videoclips([black, clip_a, black, clip_b])
#
#     txt_clip_q = TextClip(f"Q{q_num}", fontsize=40, font='Liberation-Sans', color='white')
#     txt_clip_q = txt_clip_q.set_position(("left", "top")).set_duration(18)
#     final_clip = CompositeVideoClip([final_clip, txt_clip_q])
#     return final_clip


def make_all_comp_clips(inputs, input_dir, out_dir):

    for (i, a, b, t_start, clip_len) in tqdm(inputs):
        # make
        output_clip = make_snippet(i, a, b, t_start, clip_len, input_dir)

        # save
        output_clip.write_videofile(os.path.join(out_dir, f"Q{i}.mp4"), verbose=False)
