import cv2
import random
import time

def display_message(frame, text, position, size=1, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, size, 2)[0]
    text_x = int(position[0] - text_size[0] // 2)
    text_y = int(position[1])
    cv2.putText(frame, text, (text_x, text_y), font, size, color, 2, cv2.LINE_AA)

def show_start_screen(frame, frame_w, frame_h):
    frame[:] = 0
    display_message(frame, "Press SPACE or Click to Start", (frame_w // 2, frame_h // 3), size=1.5)

def show_game_over_screen(frame, frame_w, frame_h, score):
    frame[:] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Judul GAME OVER
    cv2.putText(frame, "GAME OVER", (frame_w // 2 - 120, frame_h // 3), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # Score akhir
    cv2.putText(frame, f"Score: {score}", (frame_w // 2 - 80, frame_h // 2), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Pesan restart dan exit
    cv2.putText(frame, "Press SPACE to Restart", (frame_w // 2 - 150, int(frame_h // 1.5)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press ESC or Q to Exit", (frame_w // 2 - 140, int(frame_h // 1.3)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Tampilkan leaderboard
    try:
        with open("leaderboard.txt", "r") as f:
            scores = [line.strip() for line in f.readlines()]
        cv2.putText(frame, "Leaderboard (Top 5):", (frame_w // 2 - 150, int(frame_h // 1.15)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        for i, s in enumerate(scores):
            y_pos = int(frame_h // 1.1) + i * 30
            cv2.putText(frame, f"{i+1}. {s}", (frame_w // 2 - 100, y_pos), font, 0.8, (200, 200, 200), 2)
    except:
        pass

def generate_ground_block(frame_w, frame_h, last_blocks):
    recent = last_blocks[-3:] if len(last_blocks) >= 3 else last_blocks
    if sum(1 for b in recent if b['type'] == 'gap') >= 1:
        block_type = "platform"
    else:
        block_type = random.choices(["platform", "gap", "platform"], weights=[5, 1, 5], k=1)[0]

    block_width = random.randint(100, 140) if block_type == "platform" else random.randint(40, 60)
    block_height = 20
    gap = random.randint(50, 80)
    y = random.randint(int(frame_h * 0.7), int(frame_h * 0.8))

    if last_blocks:
        last_block = last_blocks[-1]
        x = last_block['x'] + last_block['w'] + gap
    else:
        x = frame_w + gap

    return {
        "x": int(x),
        "y": int(y),
        "w": int(block_width),
        "h": int(block_height),
        "type": block_type
    }

def reset_game(state, frame_h, frame_w, block_image, char_scale):
    state.update({
        'jumping': False,
        'y_velocity': 0,
        'score': 0,
        'previous_pitch': 0,
        'previous_volume': 0,
        'game_over': False,
        'game_started': True,
        'ground_blocks': []
    })
    y = int(frame_h * 0.65)
    for i in range(4):
        state['ground_blocks'].append({
            "x": i * 160,
            "y": int(frame_h * 0.75),
            "w": 140,
            "h": 20,
            "type": "platform"
        })
    for i in range(4, 8):
        block = generate_ground_block(frame_w, frame_h, state['ground_blocks'])
        block["x"] = i * 160
        state['ground_blocks'].append(block)

    first = next((b for b in state['ground_blocks'] if b['type'] == 'platform'), None)
    state['x'] = frame_w // 3
    state['y'] = first["y"] - int(frame_h * char_scale) if first else y
    state['game_start_time'] = time.time()
