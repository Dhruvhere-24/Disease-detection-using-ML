import svm_model_crop as svm
import cnn_model as cnn

def hybrid_model(img):
    return svm.use_svm_model(img) and cnn.predict_image(img)

if __name__ == "__main__":
    image = "test.JPG"
    result = hybrid_model(image)
    print(result)