"""
Streamlit Frontend — GCN Action Recognition Demo
=================================================
Hỗ trợ 2 nguồn đầu vào:
  • Webcam  — real-time, gọi /predict từng frame
  • Video file — batch: YOLO trích keypoints toàn bộ → /predict_batch → replay

Chạy sau khi FastAPI đã start:
    streamlit run demo/app.py
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import requests
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO

# ── Cấu hình ───────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000"
YOLO_MODEL  = str(ROOT / "yolo11m-pose.pt")
CONF_THRESH = 0.3
MAX_FPS     = 20

ACTION_COLORS = {
    "Fall Down":  (255, 80,  80),
    "Lying Down": (255, 200,  0),
    "Walking":    (80,  220, 80),
}
ACTION_EMOJI = {
    "Fall Down":  "🚨",
    "Lying Down": "🛋️",
    "Walking":    "🚶",
}
DEFAULT_COLOR = (200, 200, 200)

COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),
]

# Debounce: chỉ log khi action giữ nguyên N frame liên tiếp
LOG_DEBOUNCE_FRAMES = 10


# ── Cache resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL)


# ── API helpers ──────────────────────────────────────────────────────────────
def call_predict(api_url: str, keypoints: np.ndarray) -> dict | None:
    try:
        r = requests.post(
            f"{api_url}/predict",
            json={"frame": {"keypoints": keypoints.tolist()}},
            timeout=1.0,
        )
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


def call_predict_batch(api_url: str, all_keypoints: list) -> list | None:
    try:
        payload = {"frames": [{"keypoints": k.tolist()} for k in all_keypoints]}
        r = requests.post(f"{api_url}/predict_batch", json=payload, timeout=120.0)
        if r.status_code == 200:
            return r.json()["results"]
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi gọi API batch: {e}")
    return None


def call_reset(api_url: str):
    try:
        requests.post(f"{api_url}/reset", timeout=1.0)
    except requests.exceptions.RequestException:
        pass


def check_api_health(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=2.0)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, color=(0, 230, 120)):
    pts = [(int(k[0]), int(k[1])) for k in kpts]
    for i, j in COCO_EDGES:
        if kpts[i][2] > 0.1 and kpts[j][2] > 0.1:
            cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)
    for x, y, c in kpts:
        if c > 0.1:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 4, color, 1)


def draw_label(frame: np.ndarray, label: str, confidence: float, ready: bool):
    w = frame.shape[1]
    if not ready:
        cv2.rectangle(frame, (0, 0), (340, 42), (30, 30, 30), -1)
        cv2.putText(frame, "Warming up...", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    color = ACTION_COLORS.get(label, DEFAULT_COLOR)
    cv2.putText(frame, f"{label}  {confidence*100:.1f}%", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def draw_progress_bar(frame: np.ndarray, current: int, total: int):
    if total <= 0:
        return
    w, h = frame.shape[1], frame.shape[0]
    y = h - 6
    cv2.rectangle(frame, (0, y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, y), (int(w * current / total), h), (100, 180, 255), -1)


# ── Action Log ───────────────────────────────────────────────────────────────
def append_action_log(label: str, confidence: float,
                      timestamp_str: str | None = None) -> None:
    """Thêm 1 entry vào action log và in ra terminal."""
    if "action_log" not in st.session_state:
        st.session_state.action_log = []
    ts    = timestamp_str or datetime.now().strftime("%H:%M:%S")
    emoji = ACTION_EMOJI.get(label, "❓")

    st.session_state.action_log.append({
        "time":   ts,
        "emoji":  emoji,
        "action": label,
        "conf":   f"{confidence*100:.1f}%",
    })

    # In ra terminal Streamlit ngay lập tức
    print(f"[ACTION LOG] {ts}  {emoji} {label:<12}  conf={confidence*100:.1f}%",
          flush=True)

    # Giữ tối đa 200 entries
    if len(st.session_state.action_log) > 200:
        st.session_state.action_log = st.session_state.action_log[-200:]


def render_action_log(container) -> None:
    """Render bảng log, mới nhất trên cùng."""
    log = st.session_state.get("action_log", [])
    if not log:
        container.caption("Chưa có log. Bắt đầu để ghi nhận hành động.")
        return
    rows = list(reversed(log))
    md = "| Thời gian | Hành động | Độ tin cậy |\n|---|---|---|\n"
    for r in rows:
        md += f"| `{r['time']}` | {r['emoji']} {r['action']} | {r['conf']} |\n"
    container.markdown(md)


# ── Webcam: real-time ────────────────────────────────────────────────────────
def run_webcam(
    yolo_model, api_url, show_skeleton, yolo_conf, class_names,
    webcam_idx, frame_placeholder, status_placeholder,
    label_box, conf_box, probs_box, fps_box, log_container,
):
    call_reset(api_url)
    cap = cv2.VideoCapture(int(webcam_idx))
    if not cap.isOpened():
        frame_placeholder.error(
            f"Không mở được webcam #{webcam_idx}! "
            "Thử đổi Webcam index hoặc kiểm tra kết nối."
        )
        st.session_state.running = False
        return

    prev_time      = time.time()
    frame_count    = 0
    prev_label     = None
    debounce_count = 0

    print("[WEBCAM] Bắt đầu", flush=True)

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results  = yolo_model(rgb, conf=yolo_conf, verbose=False)
            kpts_arr = np.zeros((17, 3), dtype=np.float32)
            detected = False
            if results and results[0].keypoints is not None:
                kp = results[0].keypoints
                if kp.xy is not None and len(kp.xy) > 0:
                    kpts_arr[:, :2] = kp.xy[0].cpu().numpy()
                    kpts_arr[:, 2]  = kp.conf[0].cpu().numpy()
                    detected = True
                    if show_skeleton:
                        draw_skeleton(rgb, kpts_arr)

            result     = call_predict(api_url, kpts_arr)
            label      = result["label"]      if result else "–"
            confidence = result["confidence"] if result else 0.0
            probs      = result["probs"]      if result else []
            ready      = result["ready"]      if result else False

            draw_label(rgb, label, confidence, ready)

            # ── Log action khi đủ debounce ────────────────────────────────
            if ready and label != "–":
                if label == prev_label:
                    debounce_count += 1
                else:
                    debounce_count = 1
                    prev_label     = label
                if debounce_count == LOG_DEBOUNCE_FRAMES:
                    append_action_log(label, confidence)
                    render_action_log(log_container)

            # ── FPS ───────────────────────────────────────────────────────
            now  = time.time()
            fps  = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame_count += 1
            wait = (1.0 / MAX_FPS) - (time.time() - now)
            if wait > 0:
                time.sleep(wait)

            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            status_placeholder.caption(
                f"Frame #{frame_count} | "
                f"Person: {'✅' if detected else '❌'} | "
                f"API: {'✅' if result else '❌'}"
            )
            label_box.metric("Action", label if ready else "Warming up…")
            conf_box.metric("Confidence", f"{confidence*100:.1f}%" if ready else "–")
            if probs and ready and class_names:
                probs_box.markdown("\n".join(
                    f"**{class_names[i] if i < len(class_names) else i}**: {p*100:.1f}%"
                    for i, p in enumerate(probs)
                ))
            fps_box.metric("FPS", f"{fps:.1f}")

    finally:
        cap.release()
        st.session_state.running = False
        print("[WEBCAM] Dừng", flush=True)
        frame_placeholder.info("Camera đã tắt. Nhấn Bắt đầu để chạy lại.")


# ── Video: batch ─────────────────────────────────────────────────────────────
def run_video_batch(
    yolo_model, api_url, show_skeleton, yolo_conf, class_names,
    tmp_path, video_name, video_speed,
    frame_placeholder, status_placeholder,
    label_box, conf_box, probs_box, fps_box, log_container,
):
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        frame_placeholder.error("Không mở được file video!")
        st.session_state.running = False
        return

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur     = total / src_fps
    display_interval = max(1, video_speed)

    # ── Bước 1: YOLO extraction ──────────────────────────────────────────────
    print(f"[VIDEO] YOLO trích keypoints: {video_name} ({total} frames)", flush=True)
    status_placeholder.info(f"YOLO đang xử lý {video_name} ({total} frames)...")
    prog = st.progress(0, text="YOLO đang xử lý...")

    all_kpts, all_frames = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = yolo_model(rgb, conf=yolo_conf, verbose=False)
        kpts_arr = np.zeros((17, 3), dtype=np.float32)
        if results and results[0].keypoints is not None:
            kp = results[0].keypoints
            if kp.xy is not None and len(kp.xy) > 0:
                kpts_arr[:, :2] = kp.xy[0].cpu().numpy()
                kpts_arr[:, 2]  = kp.conf[0].cpu().numpy()
        all_kpts.append(kpts_arr)
        all_frames.append(rgb.copy())
        frame_idx += 1
        if total > 0 and frame_idx % max(1, total // 100) == 0:
            pct = frame_idx / total
            prog.progress(pct, text=f"YOLO: {frame_idx}/{total} ({pct*100:.0f}%)")

    cap.release()
    prog.progress(1.0, text=f"Đã xử lý {frame_idx} frames")

    if not all_kpts:
        status_placeholder.error("Không đọc được frame nào!")
        st.session_state.running = False
        return

    # ── Bước 2: API batch ─────────────────────────────────────────────────────
    print(f"[VIDEO] Gửi {len(all_kpts)} frames lên API...", flush=True)
    status_placeholder.info(f"Gửi {len(all_kpts)} frames lên API...")
    batch_results = call_predict_batch(api_url, all_kpts)

    if batch_results is None:
        status_placeholder.error("API batch thất bại!")
        st.session_state.running = False
        return

    prog.empty()
    status_placeholder.success(
        f"Hoàn tất! {len(batch_results)} frames | {dur:.1f}s | {src_fps:.0f} FPS gốc"
    )
    print(f"[VIDEO] Bắt đầu replay: {len(batch_results)} frames", flush=True)

    # ── Bước 3: Replay + log TỪNG FRAME (tiến dần) ───────────────────────────
    frame_delay    = max(0.01, display_interval / src_fps)
    n              = min(len(all_frames), len(batch_results))
    prev_label     = None
    debounce_count = 0

    for i in range(0, n, display_interval):
        if not st.session_state.running:
            break

        rgb    = all_frames[i].copy()
        result = batch_results[i]
        label  = result["label"]
        conf   = result["confidence"]
        ready  = result["ready"]
        probs  = result["probs"]
        kpts   = all_kpts[i]

        # ── Log tiến dần theo frame đang phát ────────────────────────────
        if ready and label != "–":
            if label == prev_label:
                debounce_count += 1
            else:
                debounce_count = 1
                prev_label     = label
            if debounce_count == LOG_DEBOUNCE_FRAMES:
                ts_sec = i / src_fps
                ts_str = f"{int(ts_sec // 60):02d}:{ts_sec % 60:05.2f}"
                append_action_log(label, conf, timestamp_str=ts_str)
                render_action_log(log_container)         # cập nhật UI ngay

        if show_skeleton:
            draw_skeleton(rgb, kpts)
        draw_label(rgb, label, conf, ready)
        draw_progress_bar(rgb, i + 1, n)

        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
        status_placeholder.caption(
            f"Frame {i+1}/{n} ({(i+1)/n*100:.0f}%) | "
            f"Person: {'✅' if kpts.any() else '❌'}"
        )
        label_box.metric("Action", label if ready else "Warming up…")
        conf_box.metric("Confidence", f"{conf*100:.1f}%" if ready else "–")
        if probs and ready and class_names:
            probs_box.markdown("\n".join(
                f"**{class_names[j] if j < len(class_names) else j}**: {p*100:.1f}%"
                for j, p in enumerate(probs)
            ))
        fps_box.metric("Tốc độ", f"{display_interval}×")
        time.sleep(frame_delay)

    st.session_state.running = False
    print("[VIDEO] Replay xong.", flush=True)
    frame_placeholder.success("Video đã phát xong!")


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="GCN Action Recognition", layout="wide")
st.title("🤸 GCN Action Recognition — Live Demo")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    api_url = st.text_input("FastAPI URL", value=API_URL).rstrip("/")
    st.caption(f"📖 Docs: [{api_url}/docs]({api_url}/docs)")

    st.divider()
    health = check_api_health(api_url)
    if health:
        st.success("✅ API đang chạy")
        st.write(f"**Classes:** {', '.join(health.get('class_names', []))}")
        st.write(f"**Device:** `{health.get('device', '?')}`")
        st.write(f"**Window:** {health.get('window_size', '?')} frames")
        class_names_cache = health.get("class_names", [])
    else:
        st.error("❌ Không kết nối được API")
        st.code("uvicorn api.main:app --port 8000", language="bash")
        class_names_cache = []

    st.divider()
    st.subheader("📥 Nguồn đầu vào")
    source_mode = st.radio("Chọn nguồn", ["📷 Webcam", "🎬 Video file"], horizontal=True)

    uploaded_video = None
    video_speed    = 1
    webcam_idx     = 0

    if source_mode == "📷 Webcam":
        webcam_idx = st.number_input(
            "Webcam index", min_value=0, max_value=5, value=0, step=1,
            help="Thường là 0. Nếu không mở được thử 1, 2..."
        )
    else:
        uploaded_video = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv", "wmv"],
        )
        video_speed = st.select_slider(
            "Tốc độ phát lại", options=[1, 2, 4, 8], value=1,
            format_func=lambda x: f"{x}×",
        )
        st.info(
            "💡 Video mode:\n"
            "1. YOLO trích keypoints toàn bộ\n"
            "2. Gửi batch → API\n"
            "3. Replay kèm log tiến dần"
        )

    st.divider()
    st.subheader("🎨 Hiển thị")
    show_skeleton = st.toggle("Hiện skeleton", value=True)
    yolo_conf     = st.slider("YOLO confidence", 0.1, 0.9, CONF_THRESH, 0.05)

    st.divider()
    run_btn  = st.button("▶ Bắt đầu", type="primary", use_container_width=True)
    stop_btn = st.button("⏹ Dừng",    use_container_width=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        reset_btn = st.button("🔄 Reset buffer", use_container_width=True)
    with col_r2:
        clear_log = st.button("🗑️ Xóa log",     use_container_width=True)

    if reset_btn:
        call_reset(api_url)
        st.toast("Buffer đã reset!")
    if clear_log:
        st.session_state.action_log = []
        st.toast("Log đã xóa!")


# ── Session state ─────────────────────────────────────────────────────────────
if "running"    not in st.session_state:
    st.session_state.running    = False
if "action_log" not in st.session_state:
    st.session_state.action_log = []
if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ── Layout chính ──────────────────────────────────────────────────────────────
col_video, col_stats = st.columns([3, 1])

with col_video:
    frame_placeholder  = st.empty()
    status_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Kết quả")
    label_box = st.empty()
    conf_box  = st.empty()
    probs_box = st.empty()
    st.divider()
    fps_box   = st.empty()

# ── Action Log (luôn hiển thị bên dưới) ──────────────────────────────────────
st.divider()
log_hdr, log_cnt = st.columns([5, 1])
with log_hdr:
    st.subheader("📋 Lịch sử hành động")
with log_cnt:
    st.metric("Entries", len(st.session_state.action_log))

log_container = st.empty()
render_action_log(log_container)

# ── Màn hình chờ / chạy ───────────────────────────────────────────────────────
if not st.session_state.running:
    if not health:
        frame_placeholder.warning(
            "⚠️ Hãy khởi động FastAPI backend trước:\n\n"
            "```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```\n\n"
            "Sau đó nhấn F5 tải lại trang, rồi nhấn ▶ Bắt đầu."
        )
    elif source_mode == "🎬 Video file" and uploaded_video is None:
        frame_placeholder.info("📂 Upload video ở sidebar rồi nhấn ▶ Bắt đầu.")
    else:
        frame_placeholder.info("👆 Nhấn ▶ Bắt đầu để bắt đầu nhận dạng hành động.")

else:
    if not health:
        frame_placeholder.error("❌ API chưa chạy! Khởi động backend trước.")
        st.session_state.running = False
    elif source_mode == "🎬 Video file" and uploaded_video is None:
        frame_placeholder.error("❌ Chưa chọn video! Upload file ở sidebar.")
        st.session_state.running = False
    else:
        yolo_model = load_yolo()

        if source_mode == "📷 Webcam":
            run_webcam(
                yolo_model, api_url, show_skeleton, yolo_conf, class_names_cache,
                webcam_idx, frame_placeholder, status_placeholder,
                label_box, conf_box, probs_box, fps_box, log_container,
            )
        else:
            suffix = Path(uploaded_video.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            try:
                run_video_batch(
                    yolo_model, api_url, show_skeleton, yolo_conf, class_names_cache,
                    tmp_path, uploaded_video.name, video_speed,
                    frame_placeholder, status_placeholder,
                    label_box, conf_box, probs_box, fps_box, log_container,
                )
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
