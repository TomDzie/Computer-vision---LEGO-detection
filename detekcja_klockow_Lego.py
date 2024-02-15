import numpy as np
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

model_path = 'best.pt'



def rescale(img, h, w):
    output_size = (int(h), int(w))
    dims = (h, w) = img.shape[:2]

    max_dim = dims.index(max(dims))
    
    dimension = [0, 0]
    dimension[max_dim] = output_size[max_dim]
    aspect = min(dims) / max(dims)
    dimension[dimension.index(min(dimension))] = int(dimension[max_dim] * aspect)
    resizedImage = cv2.resize(img, dimension[::-1], interpolation=cv2.INTER_AREA)

    resizedImage = cv2.copyMakeBorder(
    resizedImage,
    top=output_size[0] - dimension[0],
    bottom=0,
    right=0,
    left=output_size[1] - dimension[1],
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 255, 255],
    )
    return resizedImage

def segmentation(model_path, img):
    model = YOLO(model_path)
    model = model.to('cpu')
    results = model(img)
    return results


def cut_mask(img, mask):
    mask[mask > 0] = 1
    for i in range(img.shape[2]):
        img[:,:,i] = mask * img[:,:,i]
    return img



def blur(img):
    for i in range(10):
        img = cv2.medianBlur(img, 7)
    return img


def canny(img):
    img_canny = cv2.Canny(img, 20, 20)
    return(img_canny)

def find_contours(img):
    (cnt, hierarchy) = cv2.findContours( 
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    return cnt, hierarchy

def move_file(source_path, destination_path):
    try:
        shutil.move(source_path, destination_path)
        print(f"Plik przeniesiony z {source_path} do {destination_path}")
    except Exception as e:
        print(f"Błąd podczas przenoszenia pliku: {e}")


def crop_image(img, x, y, width, height, name):
    img = Image.fromarray(img.astype('uint8'))

    cropped_image = img.crop((int(x*0.9), int(y*0.9), x + int(width*1.1), y + int(height*1.1)))
    output_image_path = f"C:\\Users\\tom19\\VScode_Projects\\Computer_vision\\klocki_do_uczenia\\klocki_wyciete_surowe\klocek{name}.png"

    cropped_image.save(output_image_path)



def open_file():
    file_path = filedialog.askopenfilename(title="Wybierz plik", filetypes=[("Obrazy", "*.png;*.jpg;*.jpeg;*.gif")])
    
    if file_path:
        #Wczytujemy img
        img = cv2.imread(file_path)
        #reskalowanie img
        img = rescale(img, 640, 640)
        #segmentacjia img
        img_seg = segmentation(model_path, img)
        #lista pozniejszych bboxow
        bbox_list = []
        #pętla po wszystkich wysegmentowanych obaszarach
        for index, i in enumerate(img_seg[0].masks.data):
            #wycinamy pojedyńczy obszar z reszty zdjęcia
            img_mask = cut_mask(img.copy(), np.array(i))
            #zmiana formatu
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
            #nakładamy median blur żeby wygładzić szczegóły klocków
            img_blur = blur(img_mask.copy())
            #nakładamy filtr cannego żeby znaleść krawędzie
            img_canny = canny(img_blur)
            #wyszukujemy narysowane obrysy klocków
            cnt, hiererchy = find_contours(img_canny)
            mask = np.zeros(img.shape, np.uint8)
            cnt_px = [x.shape[0] for x in cnt]
            #dodajemy wcześniej wycięte img do listy (było potrzebne do uczenia klasyfikatora)
            bbox_list.append([img_mask])
            for j in cnt:
                #wybieramy tylko te obrysy które są nie mniejsze niż 0.5 najwiekszego obrysu 
                if j.shape[0] < (0.5 * max(cnt_px)):
                    continue
                else:
                    #tworzymy punkty do bboxow
                    x, y, w, h = cv2.boundingRect(j)
                    bbox_list[index].append([x, y, w, h])

        #rysowanie bboxow na img
        for i in bbox_list:
            for j in i[1:]:
                img = cv2.rectangle(img, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 255, 0), 2)
                print(j)

        #zmiana formatu
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
        img =Image.fromarray(img)
        # Modyfikuj obraz (np. zmniejszenie rozmiaru do 300x300)
        img = img.resize((640, 640), Image.ANTIALIAS)

        # Konwertuj obraz do formatu obsługiwanego przez Tkinter
        tk_image = ImageTk.PhotoImage(img)

        # Ustaw obraz w etykiecie
        image_label.config(image=tk_image)
        image_label.image = tk_image


# Utwórz główne okno aplikacji
app = tk.Tk()
app.title("Detekcja klocków")

# Przycisk do otwierania pliku
open_button = tk.Button(app, text="Wybierz plik", command=open_file)
open_button.pack(pady=10)

# Etykieta do wyświetlania obrazu
image_label = tk.Label(app)
image_label.pack(fill=tk.BOTH, expand=True)

# Uruchom pętlę główną
app.mainloop() 