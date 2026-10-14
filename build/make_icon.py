"""Draws assets/plaudit.ico (and a 512 px PNG): a rounded square with a blue-violet
gradient and a white sound wave, matching the settings window's app icon."""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
STOPS = [(0.0, (90, 200, 250)), (0.55, (0, 122, 255)), (1.0, (88, 86, 214))]  # #5AC8FA -> #007AFF -> #5856D6


def gradient_colour(t: float) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return STOPS[-1][1]


def draw(size: int = 1024) -> Image.Image:
    # Diagonal gradient (160 degrees, like the CSS), clipped to a rounded square.
    gradient = Image.new("RGB", (size, size))
    pixels = gradient.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = gradient_colour(min(max((0.35 * x + 0.94 * y) / (1.29 * size), 0), 1))
    mask = Image.new("L", (size, size), 0)
    inset = round(size * 0.04)
    ImageDraw.Draw(mask).rounded_rectangle((inset, inset, size - inset, size - inset), radius=round(size * 0.22), fill=255)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(gradient, (0, 0), mask)

    # Sound wave: five rounded bars, tallest in the middle.
    draw = ImageDraw.Draw(icon)
    heights = [0.20, 0.42, 0.62, 0.42, 0.24]
    bar = size * 0.075
    gap = size * 0.06
    total = len(heights) * bar + (len(heights) - 1) * gap
    left = (size - total) / 2
    for i, h in enumerate(heights):
        x = left + i * (bar + gap)
        half = size * h / 2
        draw.rounded_rectangle((x, size / 2 - half, x + bar, size / 2 + half), radius=bar / 2, fill=(255, 255, 255, 255))
    return icon


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    big = draw()
    big.resize((512, 512), Image.LANCZOS).save(ASSETS / "plaudit.png")
    sizes = [(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)]
    big.resize((256, 256), Image.LANCZOS).save(ASSETS / "plaudit.ico", sizes=sizes)
    print("wrote", ASSETS / "plaudit.ico")


if __name__ == "__main__":
    main()
