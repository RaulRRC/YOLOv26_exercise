import os
import cv2
import argparse
from ultralytics import YOLO

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "Vest",
    1: "Person",
    2: "head",
    3: "No-Vest",
    4: "hat"
}

CONF_THRESHOLD = 0.30

DIR_DATASETS = '/home/raul/Downloads/'

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def is_inside(inner_box, outer_box):

    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2

    return (
        outer_box[0] <= cx <= outer_box[2]
        and outer_box[1] <= cy <= outer_box[3]
    )


def parse_detections(results):

    detections = []

    for box in results.boxes:

        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class": CLASS_NAMES[cls_id],
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

    return detections


def evaluate_subject(subject_box, detections, subject_type="Person"):

    has_vest = False
    has_hat = False

    for det in detections:

        cls_name = det["class"]
        box = det["box"]

        if not is_inside(box, subject_box):
            continue

        if subject_type == "Person":

            if cls_name == "Vest":
                has_vest = True

            elif cls_name == "hat":
                has_hat = True

        elif subject_type == "head":

            if cls_name == "hat":
                has_hat = True

    score = 100
    missing_items = []

    if subject_type == "Person":

        if not has_vest:
            score -= 50
            missing_items.append("VEST")

        if not has_hat:
            score -= 50
            missing_items.append("HARD HAT")

    elif subject_type == "head":

        if not has_hat:
            score = 0
            missing_items.append("HARD HAT")

    score = max(score, 0)

    if score == 100:
        color = (0, 255, 0)
        status = "SAFE"

    elif score >= 50:
        color = (0, 165, 255)
        status = "WARNING"

    else:
        color = (0, 0, 255)
        status = "DANGER"

    return {
        "score": score,
        "status": status,
        "color": color,
        "missing_items": missing_items
    }


def process_frame(frame, detections):

    persons = [d for d in detections if d["class"] == "Person"]
    heads = [d for d in detections if d["class"] == "head"]

    total_score = 0
    total_subjects = 0

    # --------------------------------------------------------
    # PERSONS
    # --------------------------------------------------------

    for person in persons:

        px1, py1, px2, py2 = person["box"]

        evaluation = evaluate_subject(
            person["box"],
            detections,
            "Person"
        )

        total_score += evaluation["score"]
        total_subjects += 1

        color = evaluation["color"]

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            color,
            3
        )

        if evaluation["missing_items"]:

            label = (
                "MISSING: "
                + ", ".join(evaluation["missing_items"])
            )

        else:
            label = "SAFE"

        cv2.putText(
            frame,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # --------------------------------------------------------
    # STANDALONE HEADS
    # --------------------------------------------------------

    for head in heads:

        associated = False

        for person in persons:

            if is_inside(head["box"], person["box"]):
                associated = True
                break

        if associated:
            continue

        hx1, hy1, hx2, hy2 = head["box"]

        evaluation = evaluate_subject(
            head["box"],
            detections,
            "head"
        )

        total_score += evaluation["score"]
        total_subjects += 1

        color = evaluation["color"]

        cv2.rectangle(
            frame,
            (hx1, hy1),
            (hx2, hy2),
            color,
            3
        )

        if evaluation["missing_items"]:
            label = (
                "HEAD MISSING: "
                + ", ".join(evaluation["missing_items"])
            )
        else:
            label = "HEAD SAFE"

        cv2.putText(
            frame,
            label,
            (hx1, hy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    global_score = (
        total_score / total_subjects
        if total_subjects > 0
        else 100
    )

    cv2.putText(
        frame,
        f"GLOBAL SAFETY SCORE: {global_score:.1f}/100",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    return frame, global_score


def process_image(image_path, model):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    results = model(image)[0]

    detections = parse_detections(results)

    image, score = process_frame(
        image,
        detections
    )

    os.makedirs("Output", exist_ok=True)

    output_path = "Output/safety_output.jpg"

    cv2.imwrite(output_path, image)

    print(f"Global score: {score:.1f}")
    print(f"Saved: {output_path}")


def process_video(video_path, model):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video: {video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs("Output", exist_ok=True)

    output_path = "Output/safety_output.mp4"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)[0]

        detections = parse_detections(results)

        frame, _ = process_frame(
            frame,
            detections
        )

        writer.write(frame)

        cv2.imshow(
            "Safety Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# ENVIRONMENT EVALUATION
# ------------------------------------------------------------

def evaluar_Entorno(Entorno=0, Model=None):
    """
    Evaluate model performance on datasets with different occlusion levels.

    Entorno 0 : No occlusion
    Entorno 1 : Low occlusion
    Entorno 2 : Medium occlusion
    Entorno 3 : High occlusion
    """

    dataset_map = {
        0: DIR_DATASETS + 'construction_occlusion_none/data.yaml',
        1: DIR_DATASETS + 'Dataset_occlusion_low/data.yaml',
        2: DIR_DATASETS + 'Dataset_occlusion_medium/data.yaml',
        3: DIR_DATASETS + 'Dataset_occlusion_High/data.yaml',
    }

    if Entorno not in dataset_map:
        raise ValueError(
            f"Invalid Entorno value: {Entorno}. Must be 0, 1, 2, or 3."
        )

    dir_dataset = dataset_map[Entorno]

    # Entorno 0 uses cache='None' (string) as in the original
    cache = 'None' if Entorno == 0 else None

    metrics = Model.val(data=dir_dataset, cache=cache)

    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP75    : {metrics.box.map75:.4f}")
    print(f"  Per-class mAP50-95: {metrics.box.maps}")
    print(f"  Per-image metrics : {metrics.box.image_metrics}")

    return metrics


# ------------------------------------------------------------
# COMPLIANCE RATE
# ------------------------------------------------------------

def Tasa_cumplimiento(Video=False):
    """
    Evaluate each detected Person and verify they have both
    a vest and a hard hat.
    If a Person is not detected, evaluate whether a standalone
    head (without a hat) is visible instead.
    """
    # TODO: implement compliance-rate logic
    pass


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main(Modelo):

    Oclussion_dict = {
        0: "None",
        1: "Low",
        2: "Medium",
        3: "High"
    }

    for i in range(4):
        print(f"Testing for Occlusion level: {Oclussion_dict[i]}")
        evaluar_Entorno(Entorno=i, Model=Modelo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=False,
        help="Image or video path (optional when running evaluation mode)"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="YOLO model path"
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run occlusion environment evaluation instead of inference"
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    if args.eval:
        main(model)

    else:

        if not args.source:
            raise ValueError(
                "Provide --source when not using --eval mode."
            )

        ext = os.path.splitext(args.source)[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            process_image(args.source, model)

        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            process_video(args.source, model)

        else:
            raise ValueError(
                f"Unsupported file type: {ext}"
            )