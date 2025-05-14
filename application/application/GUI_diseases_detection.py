import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import svm_model_crop as svm
import cnn_model as cnn
import hybrid_model as hybrid


class DiseaseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩺 AI Disease Detection")
        self.root.geometry("800x500")
        self.root.configure(bg="#0b3d0b")

        self.image_path = None
        self.selected_model = tk.StringVar(value="SVM")

        # === Title ===
        title = tk.Label(root, text="AI Disease Detection Tool", font=("Helvetica", 18, "bold"),
                         bg="#0b3d0b", fg="#d4f4d4")
        title.pack(pady=20)

        # === Top Frame ===
        self.top_frame = tk.Frame(root, bg="#0b3d0b")
        self.top_frame.pack(pady=10)

        # === Upload Panel ===
        self.upload_frame = tk.LabelFrame(self.top_frame, text="Upload Image", width=300, height=250,
                                          bg="#134e13", fg="white", font=("Helvetica", 12, "bold"))
        self.upload_frame.grid(row=0, column=0, padx=20)
        self.upload_frame.grid_propagate(False)

        self.upload_canvas = tk.Canvas(self.upload_frame, width=250, height=150, bg="#e6f7ff", bd=2, relief="ridge")
        self.upload_canvas.pack(pady=10)

        self.upload_button = tk.Button(self.upload_frame, text="Browse Image", command=self.browse_image,
                                       bg="#1b5e20", fg="white", font=("Helvetica", 10, "bold"), width=20)
        self.upload_button.pack(pady=5)

        # === Model Selector & Check Button ===
        self.model_frame = tk.LabelFrame(self.top_frame, text="Model & Check", width=300, height=250,
                                         bg="#134e13", fg="white", font=("Helvetica", 12, "bold"))
        self.model_frame.grid(row=0, column=1, padx=20)
        self.model_frame.grid_propagate(False)

        model_label = tk.Label(self.model_frame, text="Select Model:", bg="#134e13", fg="white")
        model_label.pack(pady=(30, 5))

        self.model_dropdown = ttk.Combobox(self.model_frame, values=["CNN", "SVM", "Hybrid"],
                                           textvariable=self.selected_model, state="readonly", width=20)
        self.model_dropdown.pack()

        self.check_button = tk.Button(self.model_frame, text="Check for Disease", command=self.check_disease,
                                      bg="#1b5e20", fg="white", font=("Helvetica", 10, "bold"), width=20)
        self.check_button.pack(pady=20)

        # === Result Panel ===
        self.result_frame = tk.LabelFrame(root, text="Prediction Result", width=700, height=100,
                                          bg="#134e13", fg="white", font=("Helvetica", 12, "bold"))
        self.result_frame.pack(pady=20)
        self.result_frame.pack_propagate(False)

        self.result_label = tk.Label(self.result_frame, text="", font=("Helvetica", 14, "bold"),
                                     bg="#134e13", fg="white")
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            img = Image.open(self.image_path)
            img.thumbnail((250, 150))
            self.img_tk = ImageTk.PhotoImage(img)
            self.upload_canvas.delete("all")
            self.upload_canvas.create_image(125, 75, image=self.img_tk)

    def check_disease(self):
        if not self.image_path:
            self.result_label.config(text="❌ Please upload an image!", fg="red")
            return

        try:
            selected = self.selected_model.get()
            result = None

            # === Call models with raw path ===
            if selected == "CNN":
                result = cnn.predict_image(self.image_path)
            elif selected == "SVM":
                result = svm.use_svm_model(self.image_path)
            elif selected == "Hybrid":
                result = hybrid.hybrid_model(self.image_path)

            if result:
                self.result_label.config(text="✅ Disease Present", fg="red")
            else:
                self.result_label.config(text="✅ Disease Free", fg="light green")

        except Exception as e:
            self.result_label.config(text=f"❌ Error: {str(e)}", fg="orange")



if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseDetectionApp(root)
    root.mainloop()