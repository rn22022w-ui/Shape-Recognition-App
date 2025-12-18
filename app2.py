import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import cv2
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Style (dark + card-like)
# -----------------------------
def inject_css():
    st.markdown(
        """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
.big-title { font-size: 44px; font-weight: 800; margin: 10px 0 20px 0; }
.side-title { font-size: 36px; font-weight: 800; margin: 10px 0 20px 0; line-height: 1.15; }
[data-testid="stDataFrame"] { background: rgba(255,255,255,0.02); border-radius: 12px; }
div.stButton > button { width: 100%; padding: 0.9rem 1rem; border-radius: 14px; font-size: 18px; font-weight: 700; }
img { border-radius: 16px; }
button[data-baseweb="tab"] > div { font-size: 18px; font-weight: 700; }
div[data-testid="stSlider"] { padding-top: 0.2rem; padding-bottom: 0.6rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Utils
# -----------------------------
def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像の読み込みに失敗しました（PNG/JPG推奨）")
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan}
    v = np.asarray(values, dtype=float)
    return {"count": int(v.size), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def plot_density(values: List[float], title: str, xlabel: str):
    fig = plt.figure()
    ax = plt.gca()

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        ax.text(0.5, 0.5, "データがありません", ha="center", va="center")
        ax.set_axis_off()
        st.pyplot(fig)
        return

    if _HAS_SCIPY and x.size >= 3 and np.min(x) != np.max(x):
        kde = gaussian_kde(x)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = kde(xs)
        ax.plot(xs, ys)
    else:
        bins = min(30, max(5, int(np.sqrt(x.size))))
        ax.hist(x, bins=bins, density=True, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    st.pyplot(fig)


# ★ CSV / PNG Export helpers
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # Excelで開いても文字化けしにくいUTF-8-SIG
    return df.to_csv(index=False).encode("utf-8-sig")


def img_bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNGへの変換に失敗しました")
    return buf.tobytes()


# -----------------------------
# Detections
# -----------------------------
@dataclass
class CircleDet:
    x: float
    y: float
    r: float  # radius px

    @property
    def area(self) -> float:
        return math.pi * self.r * self.r

    @property
    def diameter(self) -> float:
        return 2.0 * self.r


@dataclass
class LineDet:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self) -> float:
        return float(math.hypot(self.x2 - self.x1, self.y2 - self.y1))


@dataclass
class EllipseDet:
    cx: float
    cy: float
    a: float  # semi-major px
    b: float  # semi-minor px
    angle: float

    @property
    def major(self) -> float:
        return 2.0 * self.a

    @property
    def minor(self) -> float:
        return 2.0 * self.b

    @property
    def area(self) -> float:
        return math.pi * self.a * self.b

    @property
    def ratio(self) -> float:
        if self.b == 0:
            return float("inf")
        return self.a / self.b


# 楕円の安定化：反転＋モルフォロジー
def preprocess_for_contours(gray: np.ndarray, thresh: int, invert: bool, morph_k: int) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, int(thresh), 255, flag)
    bw = cv2.medianBlur(bw, 3)

    if morph_k > 0:
        k = 2 * int(morph_k) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bw


def detect_ellipses_from_binary(bw: np.ndarray, min_area: float) -> List[EllipseDet]:
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: List[EllipseDet] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        if cnt.shape[0] < 5:
            continue
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        except Exception:
            continue

        a = float(max(MA, ma) / 2.0)
        b = float(min(MA, ma) / 2.0)
        if b <= 0:
            continue

        out.append(EllipseDet(float(cx), float(cy), a, b, float(angle)))
    return out


def detect_circles(gray: np.ndarray, dp, min_dist, p1, p2, min_r, max_r) -> List[CircleDet]:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=float(dp), minDist=float(min_dist),
        param1=float(p1), param2=float(p2),
        minRadius=int(min_r), maxRadius=int(max_r),
    )
    out: List[CircleDet] = []
    if circles is not None:
        circles = np.squeeze(circles, axis=0)
        for x, y, r in circles:
            out.append(CircleDet(float(x), float(y), float(r)))
    return out


def detect_lines(gray: np.ndarray, c1, c2, rho, theta_deg, thr, min_len, max_gap) -> List[LineDet]:
    edges = cv2.Canny(gray, int(c1), int(c2), apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges,
        rho=float(rho),
        theta=float(np.deg2rad(theta_deg)),
        threshold=int(thr),
        minLineLength=int(min_len),
        maxLineGap=int(max_gap),
    )
    out: List[LineDet] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            out.append(LineDet(int(x1), int(y1), int(x2), int(y2)))
    return out


# -----------------------------
# Drawing
# -----------------------------
def draw_overlay(
    img_bgr: np.ndarray,
    circles: List[CircleDet],
    lines: List[LineDet],
    ellipses: List[EllipseDet],
    show_guides: bool,
) -> np.ndarray:
    out = img_bgr.copy()

    for c in circles:
        cv2.circle(out, (int(c.x), int(c.y)), int(c.r), (0, 255, 0), 2)

    for ln in lines:
        cv2.line(out, (ln.x1, ln.y1), (ln.x2, ln.y2), (255, 0, 0), 2)

    for e in ellipses:
        center = (int(e.cx), int(e.cy))
        axes = (int(e.a), int(e.b))
        cv2.ellipse(out, center, axes, float(e.angle), 0, 360, (0, 255, 0), 2)

        if show_guides:
            ang = math.radians(e.angle)
            dx, dy = math.cos(ang), math.sin(ang)
            p1 = (int(e.cx - e.a * dx), int(e.cy - e.a * dy))
            p2 = (int(e.cx + e.a * dx), int(e.cy + e.a * dy))
            cv2.line(out, p1, p2, (0, 255, 0), 1)

            dx2, dy2 = -dy, dx
            q1 = (int(e.cx - e.b * dx2), int(e.cy - e.b * dy2))
            q2 = (int(e.cx + e.b * dx2), int(e.cy + e.b * dy2))
            cv2.line(out, q1, q2, (0, 255, 0), 1)

    return out


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="CellHough-like App", layout="wide")
inject_css()

st.markdown('<div class="big-title">Drag and drop file here</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Limit 200MB per file • JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

img_bgr = read_image(uploaded.getvalue())
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

tab_circle, tab_line, tab_ellipse = st.tabs(["円", "直線", "楕円"])


def make_summary_table_ellipse(ellipses: List[EllipseDet]) -> Tuple[pd.DataFrame, List[float], List[float]]:
    if len(ellipses) == 0:
        return pd.DataFrame(), [], []

    major = [e.major for e in ellipses]
    minor = [e.minor for e in ellipses]
    area  = [e.area for e in ellipses]
    ratio = [e.ratio for e in ellipses]

    rows = []
    rows.append(["楕円数", len(ellipses), len(ellipses), len(ellipses)])
    s = summarize(major); rows.append(["長径", s["mean"], s["max"], s["min"]])
    s = summarize(minor); rows.append(["短径", s["mean"], s["max"], s["min"]])
    s = summarize(area);  rows.append(["面積", s["mean"], s["max"], s["min"]])
    s = summarize(ratio); rows.append(["長/短比", s["mean"], s["max"], s["min"]])

    df = pd.DataFrame(rows, columns=["項目", "平均", "最大", "最小"])
    length_all = major + minor
    area_all = area
    return df, length_all, area_all


# ---------------- Circle tab ----------------
with tab_circle:
    left, right = st.columns([0.38, 0.62], gap="large")

    circles: List[CircleDet] = []
    circles_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    raw_len: List[float] = []
    raw_area: List[float] = []
    over = img_bgr.copy()

    with left:
        st.markdown('<div class="side-title">円検出パラメータ</div>', unsafe_allow_html=True)
        dp = st.slider("dp", 0.5, 3.0, 1.2, 0.1, key="c_dp")
        min_dist = st.slider("minDist", 1, 500, 30, 1, key="c_mindist")
        p1 = st.slider("param1", 1, 300, 120, 1, key="c_p1")
        p2 = st.slider("param2", 1, 200, 35, 1, key="c_p2")
        min_r = st.slider("minRadius", 0, 500, 5, 1, key="c_minr")
        max_r = st.slider("maxRadius (0で上限なし)", 0, 1000, 0, 1, key="c_maxr")
        show_guides = (
            st.radio("解析補助線を表示", ["オン", "オフ"], index=0, horizontal=False, key="guides_circle") == "オン"
        )

        circles = detect_circles(gray, dp, min_dist, p1, p2, min_r, max_r)

        # 検出一覧DF
        circles_df = pd.DataFrame(
            [{
                "id": i,
                "x_px": c.x,
                "y_px": c.y,
                "radius_px": c.r,
                "diameter_px": c.diameter,
                "area_px2": c.area,
            } for i, c in enumerate(circles)]
        )

        # 統計
        diam = [c.diameter for c in circles]
        area = [c.area for c in circles]
        rows = [["円数", len(circles), len(circles), len(circles)]]
        if circles:
            s = summarize(diam); rows.append(["直径", s["mean"], s["max"], s["min"]])
            s = summarize(area); rows.append(["面積", s["mean"], s["max"], s["min"]])
        summary_df = pd.DataFrame(rows, columns=["項目", "平均", "最大", "最小"])
        raw_len, raw_area = diam, area

    with right:
        over = draw_overlay(img_bgr, circles=circles, lines=[], ellipses=[], show_guides=show_guides)
        st.image(bgr_to_rgb(over), use_container_width=True)

        # PNG Export
        st.download_button(
            "PNGダウンロード（円：検出画像）",
            data=img_bgr_to_png_bytes(over),
            file_name="circles_annotated.png",
            mime="image/png",
            key="dl_circles_png",
        )

    st.subheader("円検出結果")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # CSV Export
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.download_button(
            "CSVダウンロード（円：検出一覧）",
            data=df_to_csv_bytes(circles_df),
            file_name="circles.csv",
            mime="text/csv",
            key="dl_circles_csv",
        )
    with ccol2:
        st.download_button(
            "CSVダウンロード（円：統計表）",
            data=df_to_csv_bytes(summary_df),
            file_name="circles_summary.csv",
            mime="text/csv",
            key="dl_circles_summary_csv",
        )

    if st.button("分布グラフ表示（円）", key="btn_circle"):
        c1, c2 = st.columns(2)
        with c1:
            plot_density(raw_len, "Length distribution (Circle)", "Length [px]")
        with c2:
            plot_density(raw_area, "Area distribution (Circle)", "Area [px^2]")


# ---------------- Line tab ----------------
with tab_line:
    left, right = st.columns([0.38, 0.62], gap="large")

    lines: List[LineDet] = []
    lines_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    raw_len = []
    over = img_bgr.copy()

    with left:
        st.markdown('<div class="side-title">直線検出パラメータ</div>', unsafe_allow_html=True)

        can1 = st.slider("Canny閾値1", 0, 300, 80, 1, key="l_can1")
        can2 = st.slider("Canny閾値2", 0, 300, 180, 1, key="l_can2")
        rho = st.slider("rho", 1.0, 5.0, 1.0, 0.5, key="l_rho")
        theta = st.slider("theta[deg]", 0.1, 5.0, 1.0, 0.1, key="l_theta")
        thr = st.slider("threshold", 1, 300, 80, 1, key="l_thr")
        min_len = st.slider("minLineLength", 1, 1000, 50, 1, key="l_minlen")
        max_gap = st.slider("maxLineGap", 0, 500, 10, 1, key="l_gap")

        show_guides = (
            st.radio("解析補助線を表示", ["オン", "オフ"], index=0, horizontal=False, key="guides_line") == "オン"
        )

        lines = detect_lines(gray, can1, can2, rho, theta, thr, min_len, max_gap)

        # 検出一覧DF
        lines_df = pd.DataFrame(
            [{
                "id": i,
                "x1_px": ln.x1,
                "y1_px": ln.y1,
                "x2_px": ln.x2,
                "y2_px": ln.y2,
                "length_px": ln.length,
            } for i, ln in enumerate(lines)]
        )

        lens = [ln.length for ln in lines]
        rows = [["直線数", len(lines), len(lines), len(lines)]]
        if lines:
            s = summarize(lens); rows.append(["長さ", s["mean"], s["max"], s["min"]])
        summary_df = pd.DataFrame(rows, columns=["項目", "平均", "最大", "最小"])
        raw_len = lens

    with right:
        over = draw_overlay(img_bgr, circles=[], lines=lines, ellipses=[], show_guides=show_guides)
        st.image(bgr_to_rgb(over), use_container_width=True)

        # PNG Export
        st.download_button(
            "PNGダウンロード（直線：検出画像）",
            data=img_bgr_to_png_bytes(over),
            file_name="lines_annotated.png",
            mime="image/png",
            key="dl_lines_png",
        )

    st.subheader("直線検出結果")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # CSV Export
    lcol1, lcol2 = st.columns(2)
    with lcol1:
        st.download_button(
            "CSVダウンロード（直線：検出一覧）",
            data=df_to_csv_bytes(lines_df),
            file_name="lines.csv",
            mime="text/csv",
            key="dl_lines_csv",
        )
    with lcol2:
        st.download_button(
            "CSVダウンロード（直線：統計表）",
            data=df_to_csv_bytes(summary_df),
            file_name="lines_summary.csv",
            mime="text/csv",
            key="dl_lines_summary_csv",
        )

    if st.button("分布グラフ表示（直線）", key="btn_line"):
        plot_density(raw_len, "Length distribution (Line)", "Length [px]")


# ---------------- Ellipse tab ----------------
with tab_ellipse:
    left, right = st.columns([0.38, 0.62], gap="large")

    ellipses: List[EllipseDet] = []
    ellipses_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    raw_len = []
    raw_area = []
    bw = None
    over = img_bgr.copy()

    with left:
        st.markdown('<div class="side-title">楕円検出パラメータ</div>', unsafe_allow_html=True)

        thresh = st.slider("二値化閾値", 0, 255, 127, 1, key="e_thresh")
        min_area = st.slider("最小輪郭面積", 0, 20000, 50, 10, key="e_minarea")

        invert = st.checkbox("二値化を反転（背景が白い画像向け）", value=True, key="e_invert")
        morph = st.slider("ノイズ除去（モルフォロジー）", 0, 10, 2, 1, key="e_morph")
        show_bw = st.checkbox("二値化結果を表示", value=False, key="e_showbw")

        show_guides = (
            st.radio("解析補助線を表示", ["オン", "オフ"], index=0, horizontal=False, key="guides_ellipse") == "オン"
        )

        bw = preprocess_for_contours(gray, thresh, invert=invert, morph_k=morph)
        ellipses = detect_ellipses_from_binary(bw, min_area=min_area)

        # 検出一覧DF
        ellipses_df = pd.DataFrame(
            [{
                "id": i,
                "cx_px": e.cx,
                "cy_px": e.cy,
                "semi_major_a_px": e.a,
                "semi_minor_b_px": e.b,
                "major_diameter_px": e.major,
                "minor_diameter_px": e.minor,
                "area_px2": e.area,
                "angle_deg": e.angle,
                "aspect_ratio_a_over_b": e.ratio,
            } for i, e in enumerate(ellipses)]
        )

        summary_df, raw_len, raw_area = make_summary_table_ellipse(ellipses)

    with right:
        if show_bw and bw is not None:
            st.image(bw, caption="二値化結果", clamp=True, use_container_width=True)

            # 二値化画像もPNG保存（任意）
            st.download_button(
                "PNGダウンロード（二値化画像）",
                data=img_bgr_to_png_bytes(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)),
                file_name="binary.png",
                mime="image/png",
                key="dl_bw_png",
            )

        over = draw_overlay(img_bgr, circles=[], lines=[], ellipses=ellipses, show_guides=show_guides)
        st.image(bgr_to_rgb(over), use_container_width=True)

        # PNG Export
        st.download_button(
            "PNGダウンロード（楕円：検出画像）",
            data=img_bgr_to_png_bytes(over),
            file_name="ellipses_annotated.png",
            mime="image/png",
            key="dl_ellipses_png",
        )

    st.caption("楕円検出結果")
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("楕円が検出されませんでした。二値化閾値 / 反転 / 最小輪郭面積 を調整してください。")

    # CSV Export
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.download_button(
            "CSVダウンロード（楕円：検出一覧）",
            data=df_to_csv_bytes(ellipses_df),
            file_name="ellipses.csv",
            mime="text/csv",
            key="dl_ellipses_csv",
        )
    with ecol2:
        st.download_button(
            "CSVダウンロード（楕円：統計表）",
            data=df_to_csv_bytes(summary_df),
            file_name="ellipses_summary.csv",
            mime="text/csv",
            key="dl_ellipses_summary_csv",
        )

    if st.button("分布グラフ表示（楕円）", key="btn_ellipse"):
        c1, c2 = st.columns(2)
        with c1:
            plot_density(raw_len, "Length distribution (Ellipse)", "Length [px]")
        with c2:
            plot_density(raw_area, "Area distribution (Ellipse)", "Area [px^2]")
