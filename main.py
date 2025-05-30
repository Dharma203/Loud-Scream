import cv2
import numpy as np
import pyaudio
import pygame
import time
from game_utils import reset_game, generate_ground_block

pygame.mixer.init()
background_music = pygame.mixer.Sound("resources/background_music.mp3")

dark_overlay = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(dark_overlay, (0, 0), (640, 480), (0, 0, 50), -1)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_h, frame_w = frame.shape[:2]

p = pyaudio.PyAudio()
device_index = 1

rate = 44100
chunk = 1024
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk)

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
    'highscore': 0,
    'volume_threshold': 600,
    'volume_max': 15000,
    'game_over': False,
    'game_started': False,
    'ground_blocks': [],
    'x': frame_w // 3,
    'y': int(frame_h * 0.65)
}

MIN_JUMP_VELOCITY = -15
MAX_JUMP_VELOCITY = -40

def apply_jump_effect(sprite):
    overlay = sprite.copy()
    if overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    blue_tint = np.full_like(overlay, (255, 100, 0))
    cv2.addWeighted(blue_tint, 0.4, overlay, 0.6, 0, overlay)
    return overlay

def draw_shadow(frame, x, y, width, height):
    shadow_color = (50, 50, 50)
    shadow_pos = (int(x + width // 2), int(y + height - 5))
    shadow_size = (width // 2, 10)
    cv2.ellipse(frame, shadow_pos, shadow_size, 0, 0, 360, shadow_color, -1)

def draw_game_over_screen(frame, score, highscore, frame_w, frame_h, character_img=None):
    overlay = np.zeros_like(frame)
    for i in range(frame_h):
        alpha = 0.7 * (1 - i / frame_h)
        cv2.line(overlay, (0, i), (frame_w, i), (0, 0, 0), 1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    base_scale = 3.0
    base_thickness = 5

    text = "GAME OVER"
    text_size = cv2.getTextSize(text, font, base_scale, base_thickness)[0]
    x = (frame_w - text_size[0]) // 2
    y = frame_h // 3

    for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
        cv2.putText(frame, text, (x + dx, y + dy), font, base_scale, (0,0,150), base_thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, base_scale, (0,0,255), base_thickness, cv2.LINE_AA)

    score_text = f"Score: {score}"
    score_size = cv2.getTextSize(score_text, font, 2, 4)[0]
    score_x = (frame_w - score_size[0]) // 2
    score_y = y + 100

    for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
        cv2.putText(frame, score_text, (score_x + dx, score_y + dy), font, 2, (0,100,0), 6, cv2.LINE_AA)
    cv2.putText(frame, score_text, (score_x, score_y), font, 2, (0,255,0), 4, cv2.LINE_AA)

    # Tambahkan highscore di bawah score
    highscore_text = f"Highscore: {highscore}"
    highscore_size = cv2.getTextSize(highscore_text, font, 1.5, 3)[0]
    highscore_x = (frame_w - highscore_size[0]) // 2
    highscore_y = score_y + 70

    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        cv2.putText(frame, highscore_text, (highscore_x + dx, highscore_y + dy), font, 1.5, (0, 0, 100), 3, cv2.LINE_AA)
    cv2.putText(frame, highscore_text, (highscore_x, highscore_y), font, 1.5, (255, 215, 0), 3, cv2.LINE_AA)  # Gold color

    pulse = (np.sin(time.time() * 3) + 1) / 2

    restart_text = "Press SPACE to Restart"
    exit_text = "Press ESC or Q to Exit"

    restart_size = cv2.getTextSize(restart_text, font, 1.2, 2)[0]
    exit_size = cv2.getTextSize(exit_text, font, 1.2, 2)[0]

    restart_x = (frame_w - restart_size[0]) // 2
    restart_y = highscore_y + 70

    exit_x = (frame_w - exit_size[0]) // 2
    exit_y = restart_y + 60

    restart_color = (int(0), int(255 * pulse), 0)
    exit_color = (int(255 * pulse), 0, 0)

    cv2.putText(frame, restart_text, (restart_x, restart_y), font, 1.2, restart_color, 2, cv2.LINE_AA)
    cv2.putText(frame, exit_text, (exit_x, exit_y), font, 1.2, exit_color, 2, cv2.LINE_AA)

    if character_img is not None:
        ch_h, ch_w = character_img.shape[:2]
        scale = 0.2
        resized = cv2.resize(character_img, (int(ch_w * scale), int(ch_h * scale)))
        cx = (frame_w - resized.shape[1]) // 2
        cy = exit_y + 50
        if cy + resized.shape[0] < frame_h:
            roi = frame[cy:cy+resized.shape[0], cx:cx+resized.shape[1]]
            if resized.shape[2] == 4:
                alpha_mask = resized[:, :, 3] / 255.0
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + resized[:, :, c] * alpha_mask
            else:
                roi[:] = resized

    return frame

cv2.namedWindow("Loud Scream")

clicked = [False]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked[0] = True

cv2.setMouseCallback("Loud Scream", on_mouse)

base_speed = 5
scaling_factor = 0.05

volume_buffer = []
jump_cooldown_ms = 500
last_jump_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if not state['game_started']:
        text = "Click or Press SPACE to Start"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = max(0, (frame_w - text_size[0]) // 2)
        text_y = max(30, frame_h // 3)

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Highscore: {state['highscore']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)  # Highscore di layar utama
        cv2.imshow("Loud Scream", frame)

        if key == ord(' ') or clicked[0]:
            reset_game(state, frame_h, frame_w, block_image_resized, character_scale)
            state['game_started'] = True
            background_music.play(-1)
            clicked[0] = False
        elif cv2.getWindowProperty("Loud Scream", cv2.WND_PROP_VISIBLE) < 1:
            break

        continue

    data = stream.read(chunk, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

    volume = np.mean(np.abs(audio_data))

    volume_buffer.append(volume)
    if len(volume_buffer) > 5:
        volume_buffer.pop(0)
    smooth_volume = sum(volume_buffer) / len(volume_buffer)

    current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000

    if not state['jumping'] and smooth_volume > state['volume_threshold']:
        if current_time - last_jump_time > jump_cooldown_ms:
            norm_vol = min((smooth_volume - state['volume_threshold']) / (state['volume_max'] - state['volume_threshold']), 1.0)
            norm_vol = norm_vol ** 2
            jump_velocity = MIN_JUMP_VELOCITY + norm_vol * (MAX_JUMP_VELOCITY - MIN_JUMP_VELOCITY)
            state['jumping'] = True
            state['y_velocity'] = jump_velocity
            last_jump_time = current_time
            print(f"Jump triggered! smoothed volume={smooth_volume:.2f}, y_velocity={jump_velocity}")

    state['speed'] = base_speed + (state['score'] * scaling_factor)

    next_y = state['y'] + state['y_velocity']
    character_bottom_next = next_y + character_height

    on_platform = False
    landed = False

    for block in state['ground_blocks']:
        if block["type"] == "gap":
            continue
        block_top = block["y"]
        block_left = block["x"]
        block_right = block["x"] + block["w"]

        character_left = state['x']
        character_right = state['x'] + character_width

        horizontal_collision = (character_right > block_left) and (character_left < block_right)

        if horizontal_collision and state['y'] + character_height <= block_top and character_bottom_next >= block_top:
            state['y'] = block_top - character_height
            state['jumping'] = False
            state['y_velocity'] = 0
            on_platform = True
            if not landed:
                state['score'] += 1
            landed = True
            break

    if not on_platform:
        state['y'] = next_y
        state['y_velocity'] += state['gravity']

    for block in state['ground_blocks']:
        block['x'] -= state['speed']

    if state['ground_blocks'] and state['ground_blocks'][-1]["x"] < frame_w:
        state['ground_blocks'].append(generate_ground_block(frame_w, frame_h, state['ground_blocks']))

    state['ground_blocks'] = [b for b in state['ground_blocks'] if b["x"] + b["w"] > 0]

    if state['y'] + character_height > frame_h:
        background_music.stop()
        # Update highscore jika perlu
        if state['score'] > state['highscore']:
            state['highscore'] = state['score']
        state['game_over'] = True

    for block in state['ground_blocks']:
        if block["type"] == "platform":
            resized_block = cv2.resize(block_image_resized, (int(block["w"]), int(block["h"])))
            region = frame[int(block["y"]):int(block["y"]) + int(block["h"]),
                           int(block["x"]):int(block["x"]) + int(block["w"])]
            if region.shape[:2] == resized_block.shape[:2]:
                region[:] = resized_block

    draw_shadow(frame, int(state['x']), int(state['y']), character_width, character_height)

    sprite = character_jump if state['jumping'] else character

    if state['jumping']:
        sprite = apply_jump_effect(sprite)

    y1 = max(0, int(state['y']))
    y2 = min(frame_h, int(state['y'] + character_height))
    x1 = max(0, int(state['x']))
    x2 = min(frame_w, int(state['x'] + character_width))

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

    if state['game_over']:
        frame = draw_game_over_screen(frame, state['score'], state['highscore'], frame_w, frame_h, character_img)
        cv2.imshow("Loud Scream", frame)

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

    else:
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
            cv2.putText(frame, f"Highscore: {state['highscore']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

        cv2.putText(frame, f"Volume: {smooth_volume:.2f}", (10, frame_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Loud Scream", frame)

    if cv2.getWindowProperty("Loud Scream", cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == 27 or key == ord('q'):
        break

stream.stop_stream()
stream.close()
p.terminate()
cap.release()
cv2.destroyAllWindows()
