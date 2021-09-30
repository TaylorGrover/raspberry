from extract import *
from scipy.ndimage import rotate
from PIL import ImageTk, Image
import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master = None, title = "Rotator"):
        super().__init__(master)
        self.init_data()
        self.master.title(title)
        self.createWidgets()
        self.init_bindings()

    def createWidgets(self):
        self.panel = tk.Label(self.master)
        self.panel.pack(side = "bottom", fill = "both", expand = "yes")
        self._update_image()

    def init_data(self, event = None):
        self.topleveldir = "data/greyscale/"
        self.images = []
        self.paths = []
        self.index = 0
        for directory in os.listdir(self.topleveldir):
            for i, imagefile in enumerate(os.listdir(self.topleveldir + directory)):
                self.paths.append(self.topleveldir + directory + "/" + imagefile)
                with Image.open(self.paths[i]) as img:
                    self.images.append(np.array(img))

    def set_prev(self, event = None):
        self.index -= 1
        self._update_image()

    def set_next(self, event = None):
        self.index += 1
        self._update_image()

    def rotate_right(self, event = None):
        self.images[self.index] = rotate(self.images[self.index], 90)
        #img = Image.fromarray(self.images[self.index])
        self._update_image()

    def rotate_left(self, event = None):
        self.images[self.index] = rotate(self.images[self.index], -90)
        self._update_image()

    def _update_image(self):
        self.index %= len(self.images)
        print(self.index)
        img = ImageTk.PhotoImage(Image.fromarray(self.images[self.index]).resize((256, 256), Image.BICUBIC))
        self.panel.configure(image = img)
        self.panel.image = img
    """
    <Left>: Previous image
    <Right>: Next image
    <Up>: Rotate right
    <Down>: Rotate left
    """
    def init_bindings(self):
        self.master.bind("<Left>", self.set_prev)
        self.master.bind("<Right>", self.set_next)
        self.master.bind("<Up>", self.rotate_right)
        self.master.bind("<Down>", self.rotate_left)
        
root = tk.Tk()
app = Application(master = root)
app.mainloop()
