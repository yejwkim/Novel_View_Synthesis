from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path("results/no_mask/traj_frames/step_6999")
REF_DIR = Path("data/frames_candidates/transitions")
OUT_DIR = Path("results/no_mask/groups")
GRID_DIR = Path("results/no_mask/grid")

GROUPS = {
    "group_01": [0, 1, 2, 3],
    "group_02": [28, 29, 30, 31],
    "group_03": [32, 33, 34, 35],
    "group_04": [36, 37, 38, 39],
    "group_05": [40, 41, 42, 43],
    "group_06": [44, 45, 46, 47],
    "group_07": [48, 49, 50, 51],
    "group_08": [56, 57, 58, 59],
    "group_09": [60, 61, 62, 63],
    "group_10": [64, 65, 66, 67],
    "group_11": [88, 89, 90, 91],
    "group_12": [99, 100, 101, 102],
    "group_13": [103, 104, 105, 106],
    "group_14": [107, 108, 109, 110],
    "group_15": [111, 112, 113, 114],
    "group_16": [119, 120, 121, 122],
    "group_17": [123, 124, 125, 126],
    "group_18": [127, 128, 129, 130],
    "group_19": [131, 132, 133, 134],
    "group_20": [141, 142, 143, 144],
    "group_21": [147, 148, 149, 150],
    "group_22": [151, 152, 153, 154],
    "group_23": [155, 156, 157, 158],
    "group_24": [168, 169, 170, 171],
    "group_25": [172, 173, 174, 175],
}

REFERENCE_FRAMES = {
    "group_01": "t4_00052.jpg",
    "group_02": "t1_00002.jpg",
    "group_03": "t1_00007.jpg",
    "group_04": "t1_00010.jpg",
    "group_05": "t1_00016.jpg",
    "group_06": "t1_00029.jpg",
    "group_07": "t1_00038.jpg",
    "group_08": "t2_00003.jpg",
    "group_09": "t2_00010.jpg",
    "group_10": "t2_00014.jpg",
    "group_11": "t3_00005.jpg",
    "group_12": "t3_00011.jpg",
    "group_13": "t3_00015.jpg",
    "group_14": "t3_00034.jpg",
    "group_15": "t4_00012.jpg",
    "group_16": "t1_00002.jpg",
    "group_17": "t1_00010.jpg",
    "group_18": "t3_00008.jpg",
    "group_19": "t3_00011.jpg",
    "group_20": "t4_00019.jpg",
    "group_21": "t4_00036.jpg",
    "group_22": "t4_00044.jpg",
    "group_23": "t5_00008.jpg",
    "group_24": "t5_00014.jpg",
    "group_25": "t5_00029.jpg",
}


def load_group_images(group_name: str, frame_indices: list[int]) -> list[Image.Image]:
    images = []
    for frame_idx in frame_indices:
        frame_path = SRC_DIR / f"traj_6999_{frame_idx:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(frame_path)
        images.append(Image.open(frame_path).convert("RGB"))

    pad = 8
    width, height = images[0].size
    reference_path = REF_DIR / REFERENCE_FRAMES[group_name]
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    reference_image = Image.open(reference_path).convert("RGB")
    reference_image = reference_image.resize((width, height), Image.Resampling.LANCZOS)
    return [reference_image] + images


def create_group_image(group_name: str, group_images: list[Image.Image]) -> Path:
    pad = 8
    width, height = group_images[0].size

    canvas = Image.new(
        "RGB",
        (5 * width + 6 * pad, height + 2 * pad),
        "white",
    )

    y = pad
    for col, image in enumerate(group_images):
        x = pad + col * (width + pad)
        canvas.paste(image, (x, y))

    output_path = OUT_DIR / f"{group_name}.png"
    canvas.save(output_path)
    return output_path


def create_group_grids(group_images_by_name: dict[str, list[Image.Image]]) -> list[Path]:
    if not group_images_by_name:
        raise ValueError("No group images were provided.")

    ordered_items = list(group_images_by_name.items())
    width, height = ordered_items[0][1][0].size
    pad = 12
    header_h = 24
    figure_label_h = 26
    columns = 5
    rows = 5
    column_titles = ["Reference", "NVS 1", "NVS 2", "NVS 3", "NVS 4"]
    header_font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        header_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            14,
        )
    except OSError:
        header_font = ImageFont.load_default()

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    grid_paths = []

    for grid_idx in range(0, len(ordered_items), rows):
        chunk = ordered_items[grid_idx : grid_idx + rows]
        canvas = Image.new(
            "RGB",
            (
                columns * width + (columns + 1) * pad,
                header_h + rows * height + (rows - 1) * pad + figure_label_h,
            ),
            "white",
        )

        draw = ImageDraw.Draw(canvas)
        for col, title in enumerate(column_titles):
            left = pad + col * (width + pad)
            text_bbox = draw.textbbox((0, 0), title, font=header_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = left + (width - text_width) // 2
            y = (header_h - text_height) // 2
            draw.text((x, y), title, fill=(0, 0, 0), font=header_font)

        for row, (_, group_images) in enumerate(chunk):
            for col, image in enumerate(group_images):
                x = pad + col * (width + pad)
                y = header_h + row * (height + pad)
                canvas.paste(image, (x, y))

        grid_number = grid_idx // rows + 1
        figure_label = f"Figure A{grid_number}"
        label_bbox = draw.textbbox((0, 0), figure_label, font=header_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = (canvas.width - label_width) // 2
        label_y = canvas.height - figure_label_h + (figure_label_h - label_height) // 2
        draw.text(
            (label_x, label_y),
            figure_label,
            fill=(0, 0, 0),
            font=header_font,
        )

        output_path = GRID_DIR / f"grid_{grid_number}.png"
        canvas.save(output_path)
        grid_paths.append(output_path)

    return grid_paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    group_images_by_name = {}
    group_paths = []
    for group_name, frame_indices in GROUPS.items():
        group_images = load_group_images(group_name, frame_indices)
        group_images_by_name[group_name] = group_images
        group_paths.append(create_group_image(group_name, group_images))

    print(f"Wrote {len(GROUPS)} grouped images to {OUT_DIR}")
    grid_paths = create_group_grids(group_images_by_name)
    print(f"Wrote {len(grid_paths)} grid images to {GRID_DIR}")


if __name__ == "__main__":
    main()
