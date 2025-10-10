import cv2
import os

SAVE_DIR = "images/calibration/input"
def main():    
    cap = cv2.VideoCapture(1)
    try:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Unable to receive frame")
                break
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                save_p = os.path.join(SAVE_DIR, f"{count}.png")
                cv2.imwrite(save_p, frame)
                print(f"[INFO] Saved to {save_p}.")
                count+=1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] All processes shutdown successfully!")

if __name__ == "__main__":
    main()