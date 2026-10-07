from pathlib import Path
import sys

from tracker.mcbytetrack import run_mcbytetrack


# ============================================================
# APPLICATION ROOT
# ============================================================

APP_ROOT = Path(__file__).resolve().parent


# ============================================================
# DISPLAY HEADER
# ============================================================

def show_header():

    print()
    print("=" * 70)
    print("             YOLO11m + MCByteTrack")
    print("                 VIDEO ANALYSIS")
    print("=" * 70)
    print()


# ============================================================
# GET INPUT VIDEO
# ============================================================

def get_input_video():

    while True:

        print("Enter the path of the input video.")
        print("Example:")
        print(
            r"C:\Users\HarshithaSunkara\Downloads\video.avi"
        )
        print()

        video_path = input(
            "Input video path: "
        ).strip().strip('"')

        if not video_path:

            print()
            print("ERROR: Video path cannot be empty.")
            print()

            continue

        video_path = Path(video_path)

        if not video_path.exists():

            print()
            print("ERROR: Video file not found.")
            print(f"Path: {video_path}")
            print()

            continue

        if not video_path.is_file():

            print()
            print("ERROR: The selected path is not a file.")
            print()

            continue

        return video_path


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    show_header()

    print("Application folder:")
    print(APP_ROOT)

    print()
    print("Tracker:")
    print("MCByteTrack")

    print()
    print("Model:")
    print(
        APP_ROOT / "model" / "yolo11m.pt"
    )

    print()
    print("Output folder:")
    print(
        APP_ROOT / "output"
    )

    print()
    print("-" * 70)
    print()

    # --------------------------------------------------------
    # Get input video
    # --------------------------------------------------------

    input_video = get_input_video()

    print()
    print("=" * 70)
    print("Starting MCByteTrack...")
    print("=" * 70)
    print()

    try:

        # ----------------------------------------------------
        # Run MCByteTrack
        # ----------------------------------------------------

        metrics = run_mcbytetrack(
            input_video
        )

        print()
        print("=" * 70)
        print("APPLICATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()

        print("All outputs are stored in:")

        print(
            APP_ROOT / "output"
        )

        print()

        return 0

    except KeyboardInterrupt:

        print()
        print()
        print("Application interrupted by user.")

        return 1

    except Exception as error:

        print()
        print("=" * 70)
        print("APPLICATION FAILED")
        print("=" * 70)
        print()

        print(f"Error: {error}")

        print()

        return 1


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )