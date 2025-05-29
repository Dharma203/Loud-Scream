import cv2
import numpy as np
import pyaudio
import pygame
from audio_utils import get_pitch_autocorrelation
from game_utils import reset_game, show_start_screen, show_game_over_screen, generate_ground_block

pygame.mixer.init()
background_music = pygame.mixer.Sound("resources/background_music.mp3")

dark_overlay = np.zeros((480, 640, 3), dtype=np.uint8)  # Overlay merah saat game over
cv2.rectangle(dark_overlay, (0, 0), (640, 480), (0, 0, 50), -1)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_h, frame_w = frame.shape[:2]

p = pyaudio.PyAudio()
rate = 44100
chunk = 1024
stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

character_img = cv2.imread("resources/piyik.jpg", cv2.IMREAD_UNCHANGED)
block_image = cv2.imread("resources/Block.jpg", cv2.IMREAD_UNCHANGED)

character_scale = 0.15
character = cv2.resize(character_img, (int(frame_w * character_scale), int(frame_w * character_scale)))
character_jump = character
block_image_resized = cv2.resize(block_image, (140, 20))
character_height, character_width = character.shape[:2]

score_anim = {
    'active': False,
    'scale': 1.5,
    'alpha': 1.0,
    'pos_y': 40,
    'timer': 20,
    'pulse_dir': 1,
    'color_phase': 0
}

state = {
    'speed': 5,
    'jumping': False,
    'y_velocity': 0,
    'gravity': 1,
    'score': 0,
    'previous_pitch': 0,
    'previous_volume': 0,
    'jump_threshold': 100,
    'volume_threshold': 4000,
    'game_over': False,
    'game_started': False,
    'game_start_time': None,
    'ground_blocks': [],
    'x': frame_w // 3,
    'y': int(frame_h * 0.65)
}

def apply_jump_effect(sprite):
    overlay = sprite.copy()
    if overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    blue_tint = np.full_like(overlay, (255, 100, 0))
    cv2.addWeighted(blue_tint, 0.4, overlay, 0.6, 0, overlay)
    return overlay

def draw_shadow(frame, x, y, width, height):
    shadow_color = (50, 50, 50)
    shadow_pos = (x + width // 2, y + height - 5)
    shadow_size = (width // 2, 10)
    cv2.ellipse(frame, shadow_pos, shadow_size, 0, 0, 360, shadow_color, -1)

show_start_screen(frame, frame_w, frame_h)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or cv2.getWindowProperty("ScreamGo Hero", cv2.WND_PROP_VISIBLE) < 1:
        reset_game(state, frame_h, frame_w, block_image_resized, character_scale)
        break

background_music.play(-1)

base_speed = 5
scaling_factor = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    audio_data = np.frombuffer(stream.read(chunk), dtype=np.int16)
    pitch = get_pitch_autocorrelation(audio_data, rate)
    volume = np.mean(np.abs(audio_data))

    state['speed'] = base_speed + (state['score'] * scaling_factor)
    if volume > state['volume_threshold']:
        state['speed'] += 2

    if pitch < 500 or state['jumping']:
        for block in state['ground_blocks']:
            block["x"] -= state['speed']
    elif pitch - state['previous_pitch'] > state['jump_threshold'] and not state['jumping']:
        state['jumping'] = True
        state['y_velocity'] = -15

    if state['jumping']:
        state['y'] += state['y_velocity']
        state['y_velocity'] += state['gravity']

    if not state['jumping']:
        state['y_velocity'] += state['gravity']

    if state['ground_blocks'] and state['ground_blocks'][-1]["x"] < frame_w:
        state['ground_blocks'].append(generate_ground_block(frame_w, frame_h, state['ground_blocks']))

    state['ground_blocks'] = [b for b in state['ground_blocks'] if b["x"] + b["w"] > 0]

    landed = False
    for block in state['ground_blocks']:
        if block["type"] == "gap":
            continue
        if block["x"] < state['x'] + character_width and state['x'] < block["x"] + block["w"]:
            if state['y'] + character_height >= block["y"] and state['y'] + character_height <= block["y"] + block["h"]:
                state['y'] = block["y"] - character_height
                if state['jumping']:
                    score_anim['active'] = True
                    score_anim['scale'] = 1.5
                    score_anim['alpha'] = 1.0
                    score_anim['pos_y'] = 40
                    score_anim['timer'] = 20
                    score_anim['pulse_dir'] = 1
                    score_anim['color_phase'] = 0
                state['jumping'] = False
                state['y_velocity'] = 0
                state['score'] += 1
                landed = True
                break

    if state['y'] + character_height > frame_h:
        background_music.stop()
        state['game_over'] = True

    for block in state['ground_blocks']:
        if block["type"] == "platform":
            resized_block = cv2.resize(block_image_resized, (int(block["w"]), int(block["h"])))
            region = frame[int(block["y"]):int(block["y"]) + int(block["h"]),
                           int(block["x"]):int(block["x"]) + int(block["w"])]
            if region.shape[:2] == resized_block.shape[:2]:
                region[:] = resized_block

    draw_shadow(frame, state['x'], state['y'], character_width, character_height)

    sprite = character_jump if state['jumping'] else character

    if state['jumping']:
        sprite = apply_jump_effect(sprite)

    y1 = max(0, state['y'])
    y2 = min(frame_h, state['y'] + character_height)
    x1 = max(0, state['x'])
    x2 = min(frame_w, state['x'] + character_width)

    roi = frame[y1:y2, x1:x2]
    sprite_cropped = sprite[0:(y2 - y1), 0:(x2 - x1)]

    if sprite_cropped.shape[2] == 4:
        sprite_cropped = cv2.cvtColor(sprite_cropped, cv2.COLOR_BGRA2BGR)

    if roi.shape[:2] == sprite_cropped.shape[:2] and roi.shape[2] == sprite_cropped.shape[2]:
        gray = cv2.cvtColor(sprite_cropped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(sprite_cropped, sprite_cropped, mask=mask)
        frame[y1:y2, x1:x2] = cv2.add(bg, fg)

    if score_anim['active']:
        overlay = frame.copy()
        text = f"Score: {state['score']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = 20
        text_y = int(score_anim['pos_y'])

        if score_anim['color_phase'] == 0:
            color = (0, 255, 0)
        elif score_anim['color_phase'] == 1:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.putText(overlay, text, (text_x, text_y), font, score_anim['scale'], color, 3)

        alpha = score_anim['alpha']
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        score_anim['scale'] += 0.03 * score_anim['pulse_dir']
        if score_anim['scale'] > 2.0:
            score_anim['pulse_dir'] = -1
        elif score_anim['scale'] < 1.5:
            score_anim['pulse_dir'] = 1

        score_anim['pos_y'] -= 1.5
        score_anim['alpha'] -= 0.05

        score_anim['timer'] -= 1
        if score_anim['timer'] % 5 == 0:
            score_anim['color_phase'] = (score_anim['color_phase'] + 1) % 3

        if score_anim['timer'] <= 0 or score_anim['alpha'] <= 0:
            score_anim['active'] = False
    else:
        cv2.putText(frame, f"Score: {state['score']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (10, frame_h - 30), (int(10 + volume / 20), frame_h - 10), (255, 255, 0), -1)
    cv2.putText(frame, f"Volume: {int(volume)}", (10, frame_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {int(pitch)}", (10, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("ScreamGo Hero", frame)
    state['previous_pitch'] = pitch

    if state['game_over']:
        frame = cv2.addWeighted(frame, 0.6, dark_overlay, 0.4, 0)
        show_game_over_screen(frame, frame_w, frame_h, state['score'])
        cv2.imshow("ScreamGo Hero", frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):
                reset_game(state, frame_h, frame_w, block_image_resized, character_scale)
                background_music.play(-1)
                state['game_over'] = False
                break
            elif key == 27 or key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    if cv2.getWindowProperty("ScreamGo Hero", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()
cv2.destroyAllWindows()
