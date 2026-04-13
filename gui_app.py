import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
try:
    model = tf.keras.models.load_model('digit_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model. Make sure you run train_model.py first!")
    exit()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("400x500")
        self.root.configure(bg="#f0f0f0")

        # Instruction Label
        self.label = tk.Label(root, text="Draw a digit (0-9) below", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.label.pack(pady=10)

        # Drawing Canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor="cross", bd=2, relief="groove")
        self.canvas.pack(pady=10)

        # Bind mouse movement to the draw function
        self.canvas.bind("<B1-Motion>", self.draw)

        # Hidden PIL Image (used to capture the drawing for the model)
        self.image = Image.new("L", (280, 280), "white")
        self.draw_engine = ImageDraw.Draw(self.image)

        # Buttons
        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.btn_recognize = tk.Button(button_frame, text="Recognize Digit", font=("Helvetica", 12), bg="#4CAF50", fg="white", command=self.recognize)
        self.btn_recognize.grid(row=0, column=0, padx=10)

        self.btn_clear = tk.Button(button_frame, text="Clear Canvas", font=("Helvetica", 12), bg="#f44336", fg="white", command=self.clear)
        self.btn_clear.grid(row=0, column=1, padx=10)

        # Result Label
        self.result_label = tk.Label(root, text="Prediction: None", font=("Helvetica", 18, "bold"), fg="#333333", bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8 # Brush radius
        # Draw on the visible tkinter canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        # Draw on the hidden PIL image
        self.draw_engine.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def clear(self):
        # Clear the tkinter canvas
        self.canvas.delete("all")
        # Clear the hidden PIL image
        self.image = Image.new("L", (280, 280), "white")
        self.draw_engine = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: None")

    def recognize(self):
        print("\n--- Starting Prediction ---")
        try:
            # 1. Resize the drawing
            img_resized = self.image.resize((28, 28))
            print("1. Image resized successfully.")
            
            # 2. Invert colors
            img_inverted = ImageOps.invert(img_resized)
            print("2. Image colors inverted.")
            
            # 3. Convert to numpy array and normalize (explicitly using float32)
            img_array = np.array(img_inverted, dtype=np.float32)
            img_array = img_array / 255.0
            print("3. Converted to array and normalized.")
            
            # 4. Reshape for the CNN
            img_array = img_array.reshape(1, 28, 28, 1)
            print(f"4. Array reshaped to: {img_array.shape}")
            
            # 5. Predict using the CNN
            print("5. Asking model to predict (this might take a second)...")
            prediction = model.predict(img_array)
            print(f"   Raw prediction output: {prediction}")
            
            # Extract the best guess
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            print(f"6. Success! Guessed {digit} with {confidence:.1f}% confidence.")

            # Update the UI
            self.result_label.config(text=f"Prediction: {digit} ({confidence:.1f}%)")
            
        except Exception as e:
            # If ANYTHING fails, it will print the exact error here
            print(f"\n❌ ERROR during prediction: {e}")
            self.result_label.config(text="Error! Check terminal.")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()