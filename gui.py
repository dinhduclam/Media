import ctypes
import os
from tkinter import *
from tkinter import filedialog, ttk
import cv2
import numpy as np
import scipy.signal
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image, ImageTk



ctypes.windll.shcore.SetProcessDpiAwareness(True)

root = Tk()
ttk.Style().configure("TButton", justify=CENTER)

# Global variables
gui_width = 1385
gui_height = 595
ip_file = ""
op_file = ""
original_img = None
modified_img = None
user_arg = None
popup = None
popup_input = None


root.title("Image processing")
root.minsize(gui_width, gui_height)


def set_user_arg():
    global user_arg
    user_arg = popup_input.get()
    popup.destroy()
    popup.quit()


def open_popup_slide(text):
    global popup, popup_input
    popup = Toplevel(root)
    popup.resizable(False, False)
    popup.title("Adjust")
    text_label = ttk.Label(popup, text=text, justify=LEFT)
    text_label.pack(side=TOP, anchor=W, padx=15, pady=10)
    popup_input = ttk.Scale(popup, from_=0, to=100, orient=HORIZONTAL)
    popup_input.pack(side=TOP, anchor=NW, fill=X, padx=15)
    popup_btn = ttk.Button(popup, text="OK", command=set_user_arg).pack(pady=10)
    popup.geometry(f"400x{104+text_label.winfo_reqheight()}")
    popup_input.focus()
    popup.mainloop()


def draw_before_canvas():
    global original_img, ip_file
    original_img = Image.open(ip_file)
    original_img = original_img.convert("RGB")
    width, height = before_canvas.winfo_width(), before_canvas.winfo_height()
    resized_img = original_img.resize((width, height))
    img = ImageTk.PhotoImage(resized_img)
    before_canvas.create_image(
        width/2,
        height/2,
        image=img,
        anchor="center",
    )
    before_canvas.img = img

def draw_after_canvas(mimg):
    global modified_img
    modified_img = Image.fromarray(mimg)
    width, height = after_canvas.winfo_width(), after_canvas.winfo_height()
    resized_img = modified_img.resize((width, height))
    img = ImageTk.PhotoImage(resized_img)
    after_canvas.create_image(
        width/2,
        height/2,
        image=img,
        anchor="center",
    )
    after_canvas.img = img



def load_file():
    global ip_file
    ip_file = filedialog.askopenfilename(
        title="Open an image file",
        initialdir=".",
        filetypes=[("All Image Files", "*.*")],
    )
    draw_before_canvas()


def save_file():
    global ip_file, original_img, modified_img
    file_ext = os.path.splitext(ip_file)[1][1:]
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
    )
    modified_img = modified_img.convert("RGB")
    modified_img.save(op_file)


# frames
left_frame = ttk.LabelFrame(root, text="Original Image", labelanchor=N)
left_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

middle_frame = ttk.LabelFrame(root, text="Algorithms", labelanchor=N)
middle_frame.pack(fill=BOTH, side=LEFT, padx=5, pady=10)

right_frame = ttk.LabelFrame(root, text="Modified Image", labelanchor=N)
right_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

# left frame contents
before_canvas = Canvas(left_frame, bg="white", width=512, height=512)
before_canvas.pack(expand=1)

browse_btn = ttk.Button(left_frame, text="Browse", command=load_file)
browse_btn.pack(expand=1, anchor=SW, pady=(5, 0))

# middle frame contents
algo_canvas = Canvas(middle_frame, width=200, highlightthickness=0)
scrollable_algo_frame = Frame(algo_canvas)
scrollbar = Scrollbar(
    middle_frame, orient="vertical", command=algo_canvas.yview, width=15
)
scrollbar.pack(side="right", fill="y")
algo_canvas.pack(fill=BOTH, expand=1)
algo_canvas.configure(yscrollcommand=scrollbar.set)
algo_canvas.create_window((0, 0), window=scrollable_algo_frame, anchor="nw")
scrollable_algo_frame.bind(
    "<Configure>", lambda _: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
)


# right frame contents
after_canvas = Canvas(right_frame, bg="white", width=512, height=512)
after_canvas.pack(expand=1)

save_btn = ttk.Button(right_frame, text="Save", command=save_file)
save_btn.pack(expand=1, anchor=SE, pady=(5, 0))


# algorithm fns
def equalize_histogram(img):
    hist = calculate_histogram(img)

    cumsum = np.zeros(256)
    cumsum[0] = hist[0]

    for i in range(1, 256):
        cumsum[i] = cumsum[i-1] + hist[i]

    new_img = np.zeros(img.shape, dtype="int")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = round(255 * cumsum[img[i][j]] / cumsum[255])
    return new_img

def calculate_histogram(img):
    histogram = np.zeros(256)
    for row in img:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

def show_histogram():
    img = cv2.imread(ip_file, 0)
    hist = calculate_histogram(img)
    plt.hist(hist.ravel(), bins=256, range=(0, 255), alpha=0.5, color='blue')
    plt.xlim([0, 256])
    plt.legend(['Original Image'])
    plt.show()
def show_eq():
    img = cv2.imread(ip_file, 0)
    equ_img = equalize_histogram(img)
    plt.hist(equ_img.ravel(), bins=256, range=(0, 255), alpha=0.5, color='red')
    plt.xlim([0, 256])
    plt.legend(['Equalized Image'])
    draw_after_canvas(equ_img)
    plt.show()


# algorithm btns
ttk.Button(
    scrollable_algo_frame, 
    text="Cân bằng histogram", 
    width=30, 
    command=show_eq,
).pack(expand=1, padx=5, pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Show Histogram",
    width=30,
    command=show_histogram,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Chỉnh độ sáng",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Tìm biên Sobel",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Tìm biên Laplace",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc Gauusian\n(làm mờ)",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc trung bình",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc trung vị",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc Max",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc Min",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc sắc nét\n(Sobel)",
    width=30,
    # command=,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Lọc sắc nét\n(Laplace)",
    width=30,
    # command=,
).pack(pady=2, ipady=2)


root.mainloop()
