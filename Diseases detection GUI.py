import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import svm_model_crop as smc


class DiseaseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Detection")
        self.root.geometry("600x400")
        self.root.configure(bg="#15e88d")

        self.drop_frame = tk.Canvas(self.root, bg="#e6f7ff", width=250, height=150, highlightthickness=2)
        self.drop_frame.place(x=50, y=100)
        self.drop_frame.create_rectangle(
            3, 3, 247, 147, outline="blue", width=2, dash=(5, 5)
        )

        self.drop_label = tk.Label(self.drop_frame, text="Drag and drop photo here", fg="#007acc", bg="#e6f7ff",
                                   font=("Helvetica", 10, "bold"))
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.browse_image, bg="#007acc",
                                       fg="white", font=("Helvetica", 10, "bold"), relief="raised", borderwidth=2)
        self.upload_button.place(x=125, y=270)

        self.submit_button = tk.Button(self.root, text="Check", command=self.check_disease, bg="#007acc",
                                       fg="white", font=("Helvetica", 10, "bold"), relief="raised", borderwidth=2)
        self.submit_button.place(x=125, y=320)

        
        self.result_frame = tk.Canvas(self.root, bg="#e6f7ff", width=200, height=100, highlightthickness=2)
        self.result_frame.place(x=350, y=150)
        self.result_frame.create_rectangle(
            3, 3, 197, 97, outline="blue", width=2, dash=(5, 5)
        )

        self.result_label = tk.Label(self.result_frame, text="", font=("Helvetica", 12, "bold"), bg="#e6f7ff")
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_frame.bind("<Button-1>", self.browse_image)
        self.image_path = None

    def browse_image(self, event=None):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path  # Here the image path is stored
            img = Image.open(self.image_path)
            img.thumbnail((200, 150))
            img = ImageTk.PhotoImage(img)

            
            for widget in self.drop_frame.winfo_children():
                widget.destroy()

            img_label = tk.Label(self.drop_frame, image=img, bg="#e6f7ff")
            img_label.image = img
            img_label.place(relx=0.5, rely=0.5, anchor="center")

    def check_disease(self):
        if not self.image_path:
            self.result_label.config(text="No image uploaded!", fg="black", bg="#e6f7ff")
            return

        # Pass the image to the model for analysis
        # Load your model and process the image here
        # Example:
        # model_output = your_model_function(self.image_path)
        # is_disease_present = model_output

        is_disease_present = smc.use_svm_model(self.image_path)  # Here bool function is needed to process model output

        if is_disease_present:
            self.result_label.config(text="Disease Found", fg="white", bg="red")
        else:
            self.result_label.config(text="Disease Free", fg="black", bg="light green")


root = tk.Tk()
app = DiseaseDetectionApp(root)
root.mainloop()
