from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="DreamLoop: Fine-tune YOLO on synthetic edge cases.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print("="*50)
    print("🧠 DREAMLOOP: INITIALIZING MODEL TRAINING")
    print("="*50)

    # Load the standard, pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # Train it on our synthetic Helios dataset
    results = model.train(
        data='data.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=0, # Forces it to use the DGX GPU
        project="dreamloop_training",
        name="blizzard_model"
    )

    print("\n✅ TRAINING COMPLETE")
    print("Your new model weights are saved at: ./dreamloop_training/blizzard_model/weights/best.pt")
    print("Rename this file to 'dreamloop_yolo.pt' and send it to Person B for the demo!")
    print("="*50)

if __name__ == "__main__":
    main()