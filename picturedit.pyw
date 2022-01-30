"""
Project of Mert Gülşen
Project name : picturedit
"""
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import math


class Picture:  # Picture Class for image editing

    def __init__(self, image=None):  # image attribute will be the image of the Picture instance
        if image:  # if user has entered something
            if isinstance(image, str):
                try:
                    self.image = Image.open(image)  # if input is str, try to open an image with that path
                except FileNotFoundError:
                    print("Error: Image path does not exist.")
            elif Image.isImageType(image):
                self.image = image  # if user has given an image as input, set it to image of class
        else:
            self.image = image  # if user hasn't given an input, set it to None temporarily

    def open(self, path):  # open image with path
        image_ = Image.open(path)
        self.image = image_  # set opened image to image attribute
        return image_  # return opened image

    def show(self):
        self.image.show()  # show image in a new window

    def save(self, name=None):
        if name:  # if a filename is given
            filenames = name.split(".")  # separate extension and name
            if len(filenames) == 1:  # if there is no extension
                try:
                    file_name = self.image.filename
                    fileformat = file_name.split(".")[-1]  # get extension of original image
                    self.image.save(f"{name}.{fileformat}")  # save image with given name and original extension
                except PermissionError:  # if there is no root access to save image on top of original image
                    saved = False
                    i = 1
                    file_name = self.image.filename
                    fileformat = file_name.split(".")[-1]
                    while not saved:  # try to save it with a number after the name until it saves
                        try:
                            self.image.save(f"{name} ({i}).{fileformat}")
                            saved = True
                        except PermissionError:
                            i += 1
            elif len(filenames) == 2:  # if user has given a name and an extension
                try:
                    self.image.save(f"{filenames[0]}.{filenames[1]}")  # try to save with them
                except ValueError:  # if extension error occurs
                    print(f"Unknown image extension: .{filenames[1]}")  # print error with the extension
                except PermissionError:  # if there is no root access, try to save it with a number after name
                    saved = False
                    i = 1
                    while not saved:
                        try:
                            self.image.save(f"{filenames[0]} ({i}).{filenames[1]}")
                            saved = True
                        except PermissionError:
                            i += 1

            elif len(filenames) > 2:  # if there are too many dots, print error
                print("Too many dots in a filename, use only 1")
        else:
            try:
                self.image.save(self.image.filename)  # try to save image with the original name
            except PermissionError:  # if there is no root access
                saved = False
                i = 1
                filenames = self.image.filename.split(".")
                while not saved:  # try to save it with a number after the name until it saves
                    try:
                        self.image.save(f"{filenames[0]} ({i}).{filenames[1]}")
                        saved = True
                    except PermissionError:
                        i += 1

    def resize(self, percentage):
        """
        Resizing method for picture instance

        :param percentage: Percentage for resizing by scaling. Percentage can be in range [-100, 100]
        :return: resized image
        """
        # convert image to numpy array as unsigned integer 8 type for image processing.
        img = np.array(self.image, dtype=np.uint8)
        shape = img.shape  # get image shape
        height, width = shape[0], shape[1]  # get height and width (row and column amount)
        # determine mode by shape length, if there are 3 channels it is RGB, if there is 1 channel grayscale "L"
        mode = "RGB" if len(shape) == 3 else "L"

        np.set_printoptions(suppress=True)  # avoid scientific numbers because image processing is done with integers

        # determine new height and width with given percentage
        new_height, new_width = height+(height*(percentage/100)), width+(width*(percentage/100))
        ratio = new_width/width  # ratio of new size to old size
        horizontal_pos = np.floor(np.arange(new_width)/ratio).astype("int")  # get row-wise positions for interpolation
        vertical_pos = np.floor(np.arange(new_height)/ratio).astype("int")  # get column-wise pos. for interpolation

        if mode == "L":
            horizontally = img[:, horizontal_pos]  # interpolate image in horizontal direction
            vertically = horizontally[vertical_pos, :]  # interpolate image in vertical direction
        else:  # if image is RGB, take color channels into interpolation too.
            horizontally = img[:, horizontal_pos, :]
            vertically = horizontally[vertical_pos, :, :]
        # create an image from new image array with proper mode and set it to image of class
        # image must be created as the type unsigned integer 8 because processing was done with integers.
        self.image = Image.fromarray(vertically.astype("uint8"), mode)
        return self.image  # return new resized image

    def image_filter(self, filter_name):
        """
        Image filtering method for picture instance.

        :param filter_name: Filter name for image filtering
        :return: filtered image
        """
        # a kernel name dictionary to pick a kernel from
        kernel_names = {"gaussian_blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
                        "gaussian_blur_5x5": np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                                                       [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                                       [1, 4, 6, 4, 1]]),
                        "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
                        "vertical_sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                        "bottom_sobel": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                        "distortion": np.array([[math.cos(30), math.sin(30), 0],
                                                [math.sin(-30), math.cos(30), 0], [0, 0, 1]]),
                        "box_blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9}
        kernel = np.array([])  # create empty kernel
        if filter_name in kernel_names:
            kernel = kernel_names[filter_name]  # store chosen kernel in "kernel" variable
        rgb_kernel = np.dstack((kernel, kernel, kernel))  # create an rgb kernel for rgb images
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array

        shape = img.shape  # get image shape
        img_height, img_width = shape[0], shape[1]  # get height and width (row and column amount)
        # determine mode by shape length, if there are 3 channels it is RGB, if there is 1 channel grayscale "L"
        mode = "RGB" if len(shape) == 3 else "L"
        if mode == "RGB":
            channel_count = shape[2]  # channel count is 3 if image is RGB
        else:
            channel_count = 1  # channel count is 1 if image is grayscale

        kernel_height = kernel.shape[0]  # get kernel height (row count)
        kernel_width = kernel.shape[1]  # get kernel width (column count)
        kernel_x_mid = kernel_width // 2  # get index of middle element of a kernel row
        kernel_y_mid = kernel_height // 2  # get index of middle element of a kernel column

        new_img = np.zeros(shape)  # create empty array of new image

        """
        For image convolution, instead of regular kernel convolution over every pixel of image;
        image and kernel are set for one big multiplication to reduce time complexity of code
        from O(n^2) to O(1) where n is the kernel size.
        How does it work: A pixel and it's surrounding pixels of size (nxn) are multiplied element-wise with
        the respective kernel in "full_kernel" array. Then they are summed to form a pixel of new image.
        """
        if mode == "L":  # if image is grayscale set empty image and kernel for one multiplication
            # image is set to it's height and width where every pixel is nxn kernel size
            kernel_img = np.zeros((img_height, img_width, kernel_height, kernel_width))
            full_kernel = np.zeros((img_height, img_width, kernel_height, kernel_width))

        else:  # if image is rgb set empty image and kernel for one multiplication
            # if image is rgb, there has to be another axis for image channels which is 3
            kernel_img = np.zeros((img_height, img_width, kernel_height, kernel_width, 3))
            full_kernel = np.zeros((img_height, img_width, kernel_height, kernel_width, 3))

        # every element of full image size kernel array is set to filtering kernel.
        # if image is rgb, every element is set to rgb kernel which was defined before
        full_kernel[:, :] = kernel if mode == "L" else rgb_kernel

        """
        For image filtering, while filtering the edges of image, kernel's some elements can't match with any pixel.
        To avoid this padding is added to image array.
        """
        add = kernel_width//2  # this variable is amount of padding around the image due to kernel multiplication.
        if mode == "RGB":
            # double padding is added to img sizes because there are two sides to every axis.
            padded_img = np.zeros((img_height + add*2, img_width + add*2, channel_count))
            for i in range(3):  # add padding to every channel of RGB image
                padded_img[:, :, i] = np.pad(img[:, :, i], add, mode="edge")
        else:  # add padding to grayscale image
            padded_img = np.pad(img, add, mode="edge")

        # when iterating, padded image pixels are not iterated by changing the range with "add" variable
        # when creating "kernel_img" for multiplication padded pixels are taken to consideration for multiplication
        for i in range(add, img_height + add):  # iterate through rows of image
            for j in range(add, img_width + add):  # iterate columns of image
                # these 4 variables determine the indexes of image piece respective to kernel size.
                row_start, column_start, row_end, column_end = j - kernel_x_mid, i - kernel_y_mid, j + kernel_x_mid, i + kernel_y_mid
                if mode == "RGB":
                    # set element of "kernel_image" to determined image piece.
                    kernel_img[i-add, j-add, :] = padded_img[column_start:column_end + 1, row_start:row_end + 1, :]
                else:
                    # set element of "kernel_image" to determined image piece
                    kernel_img[i-add, j-add] = padded_img[column_start:column_end + 1, row_start:row_end + 1]

        multiplied_value_array = np.multiply(kernel_img, full_kernel)  # multiply every element of size nxn
        # sum up 2nd and 3rd axis which are values of multiplication. After summation, left values are pixels of image.
        new_img = np.sum(multiplied_value_array, (2, 3))

        # after filtering, range of pixel values might be different from [0, 255]. To avoid this, values are rescaled.
        def rescale(img_arr):
            min_value = img_arr.min()
            max_value = img_arr.max()
            # (img_arr - min_value) sets ground value for pixels to zero.
            # by dividing image array elements with (max_value - min_value) image is normalised to [0, 1] range.
            return (img_arr - min_value) / (max_value - min_value)

        if mode == "RGB":
            for i in range(3):  # iterate every channel separately
                # multiply pixel values with 255 to rescale pixel values into [0,255] range.
                new_img[:, :, i] = 255 * rescale(new_img[:, :, i])
        else:
            # multiply pixel values with 255 to rescale pixel values into [0,255] range.
            new_img = 255 * rescale(new_img)
        self.image = Image.fromarray(new_img.astype("uint8"), mode)  # set image of class to filtered new image.
        return self.image

    def grayscale(self):  # method for conversion to grayscale for RGB images
        """
        Converts RGB image to grayscale.

        :return: grayscale image.
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"
        if mode == "RGB":
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate each color channel.
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # multiply color channel values with necessary weights
            new_img = gray  # set grayscale image to new image
            self.image = Image.fromarray(new_img.astype("uint8"), "L")  # set image of class to new image.
            return self.image
        else:  # if image is already grayscale, return it.
            return self.image

    def binary(self):  # method for conversion of images to black and white
        """
        Converts images to black and white.

        :return: binary color image
        """
        img = np.array(self.image, dtype=np.uint8)/255  # convert image to numpy array and normalise it into range [0,1]
        mode = "RGB" if len(img.shape) == 3 else "L"
        new_img = np.zeros((img.shape[0], img.shape[1]))  # create new empty image
        if mode == "RGB":
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate color channels
            # take average of each pixel colors for new image pixels
            # when pixel values are rounded, they will be either 1 or 0.
            # rescale them into [0, 255] range. If value is 0, pixel is black; if 255, white.
            new_img = np.round(((r+g+b)/3))*255
        else:
            new_img = np.round(img)*255
        self.image = Image.fromarray(new_img.astype("uint8"), "L")
        return self.image

    def brightness(self, percentage):  # method for adjusting brightness of images.
        """
        Adjusts brightness of images. Input percentage can be in range [-100, 100]

        :param percentage: percentage of brightness adjustment
        :return: adjusted image
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"
        if percentage >= 0:  # if percentage is positive (image will be brighter)
            # To make an image brighter, a pixel value should be increased by an amount. Maximum can be 255
            # To use percentages, difference of 255 and pixel value is used. (255-pixel)
            # For example: to increase 30 percent, %30 of (255-pixel) is added to pixel value
            new_img = img + (255-img)*(percentage/100)
        else:
            # If percentage is negative: instead of (255-pixel), pixel value is used directly to decrease it.
            new_img = img + img * (percentage / 100)
        self.image = Image.fromarray(new_img.astype("uint8"), mode)
        return self.image

    def color(self, percentage, ch):  # method for adjusting colors of RGB images.
        """
        Adjusts colors of images. Input percentage can be in range [-100, 100]. Input channel can be "r", "g" or "b".

        Logic behind this function is similar to brightness adjustment except adjustment is made on one color channel.

        :param percentage: percentage of color adjustment
        :param ch: color channel that will be adjusted. It can be "r", "g" or "b".
        :return: adjusted image.
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"
        if mode == "RGB":
            if percentage >= 0:
                if ch == "r":
                    img[:, :, 0] = img[:, :, 0] + (255 - img[:, :, 0]) * (percentage / 100)
                elif ch == "g":
                    img[:, :, 1] = img[:, :, 1] + (255 - img[:, :, 1]) * (percentage / 100)
                else:
                    img[:, :, 2] = img[:, :, 2] + (255 - img[:, :, 2]) * (percentage / 100)
            else:
                if ch == "r":
                    img[:, :, 0] = img[:, :, 0] + img[:, :, 0] * (percentage / 100)
                elif ch == "g":
                    img[:, :, 1] = img[:, :, 1] + img[:, :, 1] * (percentage / 100)
                else:
                    img[:, :, 2] = img[:, :, 2] + img[:, :, 2] * (percentage / 100)
            self.image = Image.fromarray(img.astype("uint8"), mode)
            return self.image
        else:
            return self.image

    def rotate_90(self):
        """
        Rotates images by 90 degrees in clockwise direction.

        :return: Image rotated by 90 degrees in clockwise direction.
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"

        # to rotate an image by 90 degrees in clockwise direction, image should be turned upside down and transposed.
        # Turning image upside down is achieved by reverting rows of image.
        if mode == "RGB":
            new_img = np.zeros((img.shape[1], img.shape[0], img.shape[2]))  # create new empty image array.
            img = img[::-1]  # Revert image rows.
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate color channels
            r, g, b = np.transpose(r), np.transpose(g), np.transpose(b)  # transpose image color channels
            new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = r, g, b  # place color channels to new image array.
        else:  # if grayscale
            img = img[::-1]  # revert image rows.
            new_img = np.transpose(img)  # transpose image pixels.
        self.image = Image.fromarray(new_img.astype("uint8"), mode)
        return self.image

    def rotate_counter_90(self):
        """
        Rotates images by 90 degrees in counter-clockwise direction.

        :return: Image rotated by 90 degrees in counter-clockwise direction.
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"

        # For counter-clockwise direction, image should be transposed then turned upside down.
        # Turning image upside down is achieved by reverting rows of image.
        if mode == "RGB":
            new_img = np.zeros((img.shape[1], img.shape[0], img.shape[2]))  # create new empty image array.
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate color channels
            r, g, b = np.transpose(r), np.transpose(g), np.transpose(b)  # transpose image color channels
            new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = r, g, b  # place color channels to new image array.
            new_img = new_img[::-1]  # Revert image rows.
        else:  # if grayscale
            img = np.transpose(img)  # transpose image pixels.
            new_img = img[::-1]  # revert image rows.
        self.image = Image.fromarray(new_img.astype("uint8"), mode)
        return self.image

    def vertical_flip(self):
        """
        Flips image relative to vertical axis.

        :return: Image flipped relative to vertical axis
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"

        # To flip image relative to vertical axis, row elements of every row should be reversed.
        if mode == "RGB":
            new_img = np.zeros(img.shape)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate color channels
            r, g, b = r[:, ::-1], g[:, ::-1], b[:, ::-1]  # revert row elements of every row
            new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = r, g, b  # place color channels to new image array.
        else:  # if grayscale
            new_img = img[:, ::-1]  # revert row elements of every row
        self.image = Image.fromarray(new_img.astype("uint8"), mode)
        return self.image

    def horizontal_flip(self):
        """
        Flips image relative to horizontal axis.

        :return: Image flipped relative to horizontal axis.
        """
        img = np.array(self.image, dtype=np.uint8)  # convert image to numpy array
        mode = "RGB" if len(img.shape) == 3 else "L"

        # To flip image relative to horizontal axis, rows of image should be reversed.
        if mode == "RGB":
            new_img = np.zeros(img.shape)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # separate color channels
            r, g, b = r[::-1, :], g[::-1, :], b[::-1, :]  # reverse rows of every color channel
            new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2] = r, g, b  # place color channels to new image array.
        else:  # if grayscale
            new_img = img[::-1]  # reverse image rows
        self.image = Image.fromarray(new_img.astype("uint8"), mode)
        return self.image


class ImageFrame:  # Image frame class for displaying images in application.
    """
    ImageFrame class attributes store information of images.
    """
    def __init__(self, master, img):
        self.img = img  # image
        self.original_img = None  # original image will be set once. It will be used to reset all changes.
        self.path = None  # path of image. It will store file path of image opened in application.
        self.last_img = None  # last version of image. It will be used to revert last change.
        self.img_name = None  # image name. It will store file name of image opened in application.
        self.extension = None  # extension of opened image.
        self.master = master  # Root that image canvas will be created in.

        # A canvas is created at the middle of application window. It will be used to display images.
        # Width and height of canvas is set to zero until an image is opened.
        # Border and highlight thickness is set to zero to avoid cursor position targeting borders instead of image.
        self.canvas = Canvas(self.master, width=0, height=0, bd=0, highlightthickness=0)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)  # place canvas to the middle of application window


class TopMenu:  # Top bar menu Class

    def __init__(self, master):
        self.master = master  # Root that menu will be created in.

        # ImageFrame is created within the TopMenu class because editing tools will be inside the TopMenu class.
        # give same root as TopMenu and temporarily give None as image until an image is opened.
        self.frame = ImageFrame(self.master, None)

        # main menu
        self.menu = Menu(self.master)  # create a menu widget from tkinter module
        self.master.config(menu=self.menu)  # config menu of master root as the created menu.

        # For all menus, tearoff will be false because it is not needed.

        # file menu
        self.file_menu = Menu(self.menu, tearoff=False)  # create a menu for file options.
        self.menu.add_cascade(label="File", menu=self.file_menu)  # add file menu as a cascade to main menu
        self.file_menu.add_command(label="Open Image", command=self.open_file)  # add image opening option to file menu
        self.file_menu.add_command(label="Save", command=self.save_file)  # add image saving option
        self.file_menu.add_command(label="Save As", command=self.save_as)  # add save as option
        self.file_menu.add_command(label="Exit", command=lambda: quit())  # add exiting from application option

        # edit menu
        self.edit_menu = Menu(self.menu, tearoff=False)  # create a menu for editing options.
        self.menu.add_cascade(label="Edit", menu=self.edit_menu)  # add edit menu as a cascade to main menu
        # filters
        self.filter_menu = Menu(self.edit_menu, tearoff=False)  # create a menu for filtering options.
        self.edit_menu.add_cascade(label="Filter", menu=self.filter_menu)  # add filter menu as a cascade to edit menu
        self.filter_menu.add_command(label="Grayscale", command=self.grayscale)  # add grayscale option to edit menu
        self.filter_menu.add_command(label="Black and White", command=self.binary)  # add binary option
        self.filter_menu.add_command(label="Emboss", command=self.emboss)  # add emboss filter option
        self.filter_menu.add_command(label="Box Blur", command=self.box_blur)  # add box blur filter option
        self.filter_menu.add_command(label="Gaussian Blur", command=self.gaussian_blur)  # add gaussian blur option
        self.filter_menu.add_command(label="Gaussian Blur 5x5", command=self.gaussian_blur_5x5)  # add g_blur_5x5 option
        self.filter_menu.add_command(label="Vertical Sobel", command=self.vertical_sobel)  # add vertical sobel filter
        self.filter_menu.add_command(label="Bottom Sobel", command=self.bottom_sobel)  # add bottom sobel filter option
        self.filter_menu.add_command(label="Distortion", command=self.distortion)  # add distortion filter option
        # adjust
        self.adjust_menu = Menu(self.edit_menu, tearoff=False)  # create a menu for adjusting options.
        self.edit_menu.add_cascade(label="Adjust", menu=self.adjust_menu)  # add adjust menu as a cascade to edit menu
        self.adjust_menu.add_command(label="Brightness", command=self.brightness_window)  # add brightness option
        self.adjust_menu.add_command(label="Color", command=self.color_window)  # add color option
        # rotate
        self.rotate_menu = Menu(self.edit_menu, tearoff=False)  # create a menu for rotating options.
        self.edit_menu.add_cascade(label="Rotate", menu=self.rotate_menu)  # add rotate menu as a cascade to edit menu
        self.rotate_menu.add_command(label="Clockwise 90", command=self.rotate_90)  # add clockwise 90 option
        self.rotate_menu.add_command(label="Counter-clockwise 90", command=self.rotate_counter_90)  # add c_c_90 option
        self.rotate_menu.add_command(label="Vertical flip", command=self.vertical_flip)  # add vertical flip option
        self.rotate_menu.add_command(label="Horizontal flip", command=self.horizontal_flip)  # add horizon. flip option
        # draw
        self.edit_menu.add_command(label="Draw", command=self.draw)  # add draw option to edit menu
        # resize
        self.edit_menu.add_command(label="Resize", command=self.resize_window)  # add resizing option to edit menu

        # view menu
        self.view_menu = Menu(self.menu, tearoff=False)  # create a menu for view options.
        self.menu.add_cascade(label="View", menu=self.view_menu)  # add view menu as a cascade to main menu
        # themes
        self.theme_menu = Menu(self.view_menu, tearoff=False)  # create a menu for theme options.
        self.view_menu.add_cascade(label="Theme", menu=self.theme_menu)  # add theme menu as a cascade to view menu
        self.theme_menu.add_command(label="Dark", command=lambda: self.master.config(bg="dark grey"))  # dark grey
        self.theme_menu.add_command(label="Light", command=lambda: self.master.config(bg="light grey"))  # light grey
        self.theme_menu.add_command(label="Default", command=lambda: self.master.config(bg="SystemButtonFace"))  # def.

        # clear menu
        self.clear_menu = Menu(self.menu, tearoff=False)  # create a menu for clearing options
        self.menu.add_cascade(label="Clear", menu=self.clear_menu)  # add clear menu as a cascade to main menu
        self.clear_menu.add_command(label="Undo", command=self.undo)  # add undo option
        self.clear_menu.add_command(label="Reset", command=self.reset)  # add reset option

        #  Create a status bar with given arguments.
        #  Initial status text is waiting for user.
        self.status = Label(self.master, text=f"Waiting for user to open an image", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)  # pack status bar to bottom of window end fill horizontally.

        # bind position method to ImageFrame canvas for tracking position of cursor on the displayed image.
        # <Motion> event tracks position of mouse on bind widget.
        self.frame.canvas.bind("<Motion>", self.position)

    def position(self, event):  # position method for tracking position of mouse on displayed image.
        # position of mouse in vertical axis corresponds to row of image.
        # position of mouse in horizontal axis corresponds to column of image.
        row, col = event.y, event.x  # get row and col position of mouse on image.
        self.status["text"] = f"Cursor position : {row, col}"  # update status text with position of mouse on image.
        # if there is no image, there will be no canvas to track because initial width and height was set as zero.

    def open_file(self):  # open file method for opening images from target folders.
        global tk_img  # tk_img is made a global variable to make it possible to pass it outside of current scope.

        file_path = filedialog.askopenfilename()  # ask user for a file path with filedialog module from tkinter
        img = Picture().open(file_path)  # open image with given file path
        x, y = img.size  # get image sizes

        window_width = self.master.winfo_width()  # get application window width
        window_height = self.master.winfo_height()  # get application window height

        if x > window_width or y > window_height:  # if any of opened image sizes bigger than window sizes
            message = "Image you have opened is bigger than the size of the application, do you want to resize it ?"
            msg_box = messagebox.askquestion("Image size bigger than program", message)  # ask user for resizing

            if msg_box == "yes":  # if user accepts resizing resize it to fit current window size
                percentage_x = ((x - window_width) / (-1*x)) * 100  # get percentage for fitting width
                percentage_y = ((y - window_height) / (-1*y)) * 100  # get percentage for fitting height
                # Choose bigger percentage for resizing. Consequently, other side will fit no matter what
                percentage = percentage_x if percentage_x <= percentage_y else percentage_y

                img = Picture(img).resize(percentage)  # resize image with chosen percentage
                x, y = img.size  # get new image sizes as x and y

            else:  # if user doesn't accept resizing image,
                screen_width = self.master.winfo_screenwidth()  # get screen width
                screen_height = self.master.winfo_screenheight()  # get screen height
                if x > screen_width or y > screen_height:  # if image is bigger than screen,
                    percentage_x = ((x - screen_width) / (-1 * x)) * 100
                    percentage_y = ((y - screen_height) / (-1 * y)) * 100
                    percentage = percentage_x if percentage_x <= percentage_y else percentage_y
                    img = Picture(img).resize(percentage)  # resize image to fit into screen
                    x, y = img.size  # get new image sizes as x and y

        # convert image to tkinter PhotoImage type. Canvas can only create images with type PhotoImage.
        tk_img = ImageTk.PhotoImage(img)
        self.frame.img = img  # set image of ImageFrame to opened image
        self.frame.original_img = img  # set original image to opened image
        self.frame.path = file_path  # store opened file path for later use
        self.frame.img_name = file_path.split("/")[-1]  # store file name for later use
        self.frame.extension = self.frame.img_name.split(".")[-1]  # store file extension for later use
        self.frame.canvas.config(width=x, height=y)  # config canvas sizes with image sizes
        self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)  # create image at the middle of canvas
        self.status["text"] = "Cursor position"  # Until cursor goes on the image, status text will be "Cursor Position"

    def save_file(self):  # save image on top of original image path
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            self.frame.img.save(self.frame.path)  # save image of ImageFrame to original file path

    def save_as(self):  # save image to a new path
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            # Ask user a filepath with a name for file. Extension will be the original extension of image.
            folder_path = filedialog.asksaveasfilename(defaultextension=self.frame.extension)
            self.frame.img.save(folder_path)  # save image to target folder

    def undo(self):  # undo the last change on image
        if (self.frame.img is not None) and (self.frame.last_img is None):  # If no change has been made, give error
            messagebox.showwarning("Error", "There is no need to undo.")
            print(self.frame.img)
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img  # tk_img is made a global variable to make it possible to pass it outside of current scope.
            last_img = self.frame.last_img  # get last version of image from ImageFrame
            x, y = last_img.size  # get image sizes
            tk_img = ImageTk.PhotoImage(last_img)  # convert image to PhotoImage type
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = last_img  # set current image to last version
            self.frame.canvas.config(width=x, height=y)  # config canvas sizes with image sizes
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)  # create image at center of canvas

    def reset(self):  # clear all changes and reset to original image.
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img  # tk_img is made a global variable to make it possible to pass it outside of current scope.
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = self.frame.original_img  # set ImageFrame image to original image for resetting
            original_img = self.frame.original_img  # get original image
            x, y = original_img.size  # get original image sizes
            tk_img = ImageTk.PhotoImage(original_img)  # create a new image as PhotoImage type with original image
            self.frame.canvas.config(width=x, height=y)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def grayscale(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img  # tk_img is made a global variable to make it possible to pass it outside of current scope.
            filtered = Picture(self.frame.img).grayscale()
            x, y = filtered.size  # get new image sizes
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)

    def binary(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).binary()  # convert image to black and white with binary() of Picture()
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def emboss(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            sharpened = Picture(self.frame.img).image_filter("emboss")  # filter image with emboss filter
            x, y = sharpened.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = sharpened  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(sharpened)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def box_blur(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("box_blur")
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def gaussian_blur(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("gaussian_blur")  # filter image with gaussian blur
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def gaussian_blur_5x5(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("gaussian_blur_5x5")  # filter image with gaussian blur 5x5
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def vertical_sobel(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("vertical_sobel")  # filter image with bottom sobel operator
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def bottom_sobel(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("bottom_sobel")  # filter image with bottom sobel operator
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def distortion(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            filtered = Picture(self.frame.img).image_filter("distortion")  # filter image with distortion filter
            x, y = filtered.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = filtered  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(filtered)
            self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)

    def rotate_90(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            rotated = Picture(self.frame.img).rotate_90()  # rotate image by 90 degrees in clockwise direction
            x, y = rotated.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = rotated  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(rotated)
            self.frame.canvas.config(width=x, height=y)
            self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)

    def rotate_counter_90(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            rotated = Picture(self.frame.img).rotate_counter_90()  # rotate image by 90 degrees in c_clockwise direction
            x, y = rotated.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = rotated  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(rotated)
            self.frame.canvas.config(width=x, height=y)
            self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)

    def vertical_flip(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            flipped = Picture(self.frame.img).vertical_flip()  # flip image relative to vertical axis
            x, y = flipped.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = flipped  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(flipped)
            self.frame.canvas.config(width=x, height=y)
            self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)

    def horizontal_flip(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global tk_img
            flipped = Picture(self.frame.img).horizontal_flip()  # flip image relative to horizontal axis
            x, y = flipped.size
            self.frame.last_img = self.frame.img  # set last version as current version
            self.frame.img = flipped  # set ImageFrame image to new image
            tk_img = ImageTk.PhotoImage(flipped)
            self.frame.canvas.config(width=x, height=y)
            self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)

    def brightness_window(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            # slider_set() function wraps a slider with buttons and places it into given frame in a window.
            def slider_set(slider, frame, slider_name):  # function for placing slider onto adjustment window
                label1 = Label(frame, text=slider_name)  # label of slider
                # Increase or decrease buttons. These buttons change slider value with requested value.
                # Example: If +5 button is pressed, first it gets slider value, then it adds 5 to it and sets it.
                # Note: If buttons exceed the current value of slider, they will not do anything
                decrement_five_button = Button(frame, text="-5", command=lambda: slider.set(
                    slider.get() - 5) if slider.get() >= -95 else slider.set(slider.get()))
                decrement_one_button = Button(frame, text="-1", command=lambda: slider.set(
                    slider.get() - 1) if slider.get() >= -99 else slider.set(slider.get()))
                increment_one_button = Button(frame, text="+1", command=lambda: slider.set(
                    slider.get() + 1) if slider.get() <= 99 else slider.set(slider.get()))
                increment_five_button = Button(frame, text="+5", command=lambda: slider.set(
                    slider.get() + 5) if slider.get() <= 95 else slider.set(slider.get()))
                label1.pack()
                # when all buttons and sliders are packed to the left, they will all be in same row.
                decrement_five_button.pack(side=LEFT)  # pack -5 button to the left
                decrement_one_button.pack(side=LEFT)  # pack -1 button to the left
                slider.pack(side=LEFT)  # pack slider to the left
                increment_one_button.pack(side=LEFT)  # pack +1 button to the left
                increment_five_button.pack(side=LEFT)  # pack +5 button to the left

            def apply_changes():  # apply changes when apply button is pressed
                global tk_img
                value = slider1.get()  # get slider value
                slider1.set(0)  # reset slider value to zero
                adjusted = Picture(self.frame.img).brightness(value)  # adjust brightness of image
                x, y = adjusted.size  # get new adjusted image sizes
                self.frame.last_img = self.frame.img  # set last version as current version
                self.frame.img = adjusted  # set ImageFrame image to new image
                tk_img = ImageTk.PhotoImage(adjusted)
                self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)
                adjust_window.destroy()  # finally destroy brightness adjustment window

            adjust_window = Toplevel(self.master)  # create adjust window as a TopLevel widget
            adjust_window.title("Brightness")  # set title
            adjust_window.geometry("250x150")  # set size of window with geometry()

            # Master root of frames inside adjust window should be adjust_window.
            frame1 = Frame(adjust_window)  # frame1 will be top frame which will contain slider.
            frame2 = Frame(adjust_window)  # frame2 will be bottom frame which will contain apply and close buttons

            # Master root of slider and buttons inside are frame1 and frame2, which are slaves of adjust_window
            # Slider value can be in range -100 to 100 because brightness can be decreased by %100 or increased by %100
            slider1 = Scale(frame1, from_=-100, to=100, orient=HORIZONTAL)  # create horizontal slider
            apply_button = Button(frame2, text="Apply", command=apply_changes)  # create apply button
            close_button = Button(frame2, text="Close", command=adjust_window.destroy)  # create close button

            frame1.pack(side=TOP)  # pack frame1 to the top of adjust window
            frame2.pack(side=BOTTOM)  # pack frame2 to the bottom of adjust window

            slider_set(slider1, frame1, "Brightness")  # prepare slider with slider_set() function defined before.
            apply_button.pack(side=LEFT, padx=10, pady=10)  # pack buttons to the left with some padding around them.
            close_button.pack(side=LEFT, padx=10, pady=10)

    def resize_window(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            # slider_set() function which was explained in brightness_window() method
            # slider_set() function wraps a slider with buttons and places it into given frame in a window.
            def slider_set(slider, frame, slider_name):
                label1 = Label(frame, text=slider_name)
                decrement_five_button = Button(frame, text="-5", command=lambda: slider.set(
                    slider.get() - 5) if slider.get() >= -95 else slider.set(slider.get()))
                decrement_one_button = Button(frame, text="-1", command=lambda: slider.set(
                    slider.get() - 1) if slider.get() >= -99 else slider.set(slider.get()))
                increment_one_button = Button(frame, text="+1", command=lambda: slider.set(
                    slider.get() + 1) if slider.get() <= 99 else slider.set(slider.get()))
                increment_five_button = Button(frame, text="+5", command=lambda: slider.set(
                    slider.get() + 5) if slider.get() <= 95 else slider.set(slider.get()))
                label1.pack()
                decrement_five_button.pack(side=LEFT)
                decrement_one_button.pack(side=LEFT)
                slider.pack(side=LEFT)
                increment_one_button.pack(side=LEFT)
                increment_five_button.pack(side=LEFT)

            def apply_changes():
                global tk_img
                value = slider1.get()  # get current slider value
                slider1.set(0)  # reset slider value to zero
                resized = Picture(self.frame.img).resize(value)  # resize image with slider value as percentage
                x, y = resized.size  # get resized image sizes
                self.frame.last_img = self.frame.img  # set last version as current version
                self.frame.img = resized  # set ImageFrame image to new image
                tk_img = ImageTk.PhotoImage(resized)
                self.frame.canvas.config(width=x, height=y)  # config canvas size with new resized image sizes
                self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)
                adjust_window.destroy()  # finally destroy resizing adjustment window

            adjust_window = Toplevel(self.master)  # create adjust window as a TopLevel widget
            adjust_window.title("Resize")  # set title
            adjust_window.geometry("250x150")  # set window sizes
            frame1 = Frame(adjust_window)  # frame1 will be top frame which will contain slider.
            frame2 = Frame(adjust_window)  # frame2 will be bottom frame which will contain apply and close buttons

            slider1 = Scale(frame1, from_=-100, to=100, orient=HORIZONTAL)
            apply_button = Button(frame2, text="Apply", command=apply_changes)
            close_button = Button(frame2, text="Close", command=adjust_window.destroy)

            frame1.pack(side=TOP)  # pack frame1 to the top of adjust window
            frame2.pack(side=BOTTOM)  # pack frame2 to the bottom of adjust window

            slider_set(slider1, frame1, "Resize")  # wrap and place slider
            apply_button.pack(side=LEFT, padx=10, pady=10)  # place buttons
            close_button.pack(side=LEFT, padx=10, pady=10)

    def color_window(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            # slider_set() function which was explained in brightness_window() method
            # slider_set() function wraps a slider with buttons and places it into given frame in a window.
            def slider_set(slider, frame, slider_name):
                label1 = Label(frame, text=slider_name)
                decrement_five_button = Button(frame, text="-5", command=lambda: slider.set(
                    slider.get() - 5) if slider.get() >= -95 else slider.set(slider.get()))
                decrement_one_button = Button(frame, text="-1", command=lambda: slider.set(
                    slider.get() - 1) if slider.get() >= -99 else slider.set(slider.get()))
                increment_one_button = Button(frame, text="+1", command=lambda: slider.set(
                    slider.get() + 1) if slider.get() <= 99 else slider.set(slider.get()))
                increment_five_button = Button(frame, text="+5", command=lambda: slider.set(
                    slider.get() + 5) if slider.get() <= 95 else slider.set(slider.get()))
                label1.pack()
                decrement_five_button.pack(side=LEFT)
                decrement_one_button.pack(side=LEFT)
                slider.pack(side=LEFT)
                increment_one_button.pack(side=LEFT)
                increment_five_button.pack(side=LEFT)

            def apply_changes():
                global tk_img
                red, green, blue = slider1.get(), slider2.get(), slider3.get()  # get color percentages
                slider1.set(0), slider2.set(0), slider3.set(0)  # reset sliders to zero

                r_adjusted = Picture(self.frame.img).color(red, "r")  # adjust color channels with color() of Picture()
                g_adjusted = Picture(r_adjusted).color(green, "g")
                b_adjusted = Picture(g_adjusted).color(blue, "b")

                x, y = b_adjusted.size
                self.frame.last_img = self.frame.img  # set last version as current version
                self.frame.img = b_adjusted  # set ImageFrame image to new image
                tk_img = ImageTk.PhotoImage(b_adjusted)
                self.frame.canvas.create_image(x/2, y/2, anchor=CENTER, image=tk_img)
                adjust_window.destroy()

            adjust_window = Toplevel(self.master)  # create adjust window as a TopLevel widget
            adjust_window.title("Color")  # set title
            adjust_window.geometry("250x250")  # set window sizes
            frame1 = Frame(adjust_window)  # frame1 will contain red color slider.
            frame2 = Frame(adjust_window)  # frame2 will contain green color slider.
            frame3 = Frame(adjust_window)  # frame3 will contain blue color slider.
            frame4 = Frame(adjust_window)  # frame4 will contain apply and close buttons

            slider1 = Scale(frame1, from_=-100, to=100, orient=HORIZONTAL)  # create horizontal color sliders
            slider2 = Scale(frame2, from_=-100, to=100, orient=HORIZONTAL)
            slider3 = Scale(frame3, from_=-100, to=100, orient=HORIZONTAL)

            apply_button = Button(frame4, text="Apply", command=apply_changes)  # create buttons
            close_button = Button(frame4, text="Close", command=adjust_window.destroy)

            frame1.pack()  # pack frames in order from top to bottom.
            frame2.pack()
            frame3.pack()
            frame4.pack()
            slider_set(slider1, frame1, "Red")  # wrap color sliders
            slider_set(slider2, frame2, "Green")
            slider_set(slider3, frame3, "Blue")

            apply_button.pack(side=LEFT, padx=10, pady=10)  # pack buttons
            close_button.pack(side=LEFT, padx=10, pady=10)

    def draw(self):
        if self.frame.img is None:  # if user hasn't opened an image, give error with tkinter messagebox warning
            messagebox.showwarning("Error", "Open an image first")
        else:
            global drawing_color  # make drawing_color global to make it reachable from outer scope
            color_dict = {"black": [0, 0, 0],  # different color options
                          "red": [255, 0, 0],
                          "green": [0, 255, 0],
                          "blue": [0, 0, 255]}
            self.frame.last_img = self.frame.img  # set last version as current version
            draw_window = Toplevel(self.master)  # drawing window
            draw_window.title("Drawing")
            draw_window.geometry("250x150")
            drawing_color = "black"  # default drawing color
            x, y = self.frame.img.size  # get image sizes
            if self.frame.img.mode == "L":  # convert grayscale to rgb for drawing
                arr = np.array(self.frame.img)  # convert image to array
                new_arr = np.zeros((y, x, 3))  # create new empty image (y: height or rows, x: width or columns)
                for i in range(len(arr)):
                    for j in range(len(arr[i])):
                        # duplicate each grayscale pixel value 3 times for 3 color channels
                        new_arr[i, j] = np.array([arr[i, j] for _ in range(3)])
                self.frame.img = Image.fromarray(new_arr.astype("uint8"), "RGB")  # set ImageFrame image to new image
            self.frame.canvas.config(cursor="pencil")  # set img canvas cursor icon to pencil for drawing

            def start_drawing(event):  # catches left mouse button click event
                global func_id1  # make func_id1 global to make it reachable from end_drawing() function
                # bind draw_motion() to canvas to start drawing
                self.frame.canvas.bind("<Motion>", draw_motion)  # catch mouse position and send it to draw_motion()

            def draw_motion(event):
                global tk_img
                row, column = event.y, event.x  # get row index and column index
                self.status["text"] = f"Cursor position : {row, column}"  # update status bar cursor position text
                img = np.array(self.frame.img)  # convert image to array
                if (row in range(len(img))) and (column in range(len(img[row]))):  # if row and column index is in image
                    try:
                        # paint 3x3 area around given pixel index with current drawing color.
                        # For example: If color is red, set painted pixels to [255, 0, 0]
                        img[row-1:row+2, column-1:column+2] = color_dict[drawing_color]  # get drawing_color pixel value
                        drawn = Image.fromarray(img.astype("uint8"), "RGB")
                        self.frame.img = drawn
                        tk_img = ImageTk.PhotoImage(drawn)
                        self.frame.canvas.create_image(x / 2, y / 2, anchor=CENTER, image=tk_img)
                    except:  # If any error occurs like index out of image range when drawing close to edges, ignore it
                        pass
                else:  # if index is not in range of image rows and columns, don't perform drawing operations
                    pass

            def end_drawing(event):  # Catches left mouse button release event
                self.frame.canvas.unbind("<Motion>")  # unbind <Motion> event for ending drawing_motion()
                self.frame.canvas.bind("<Motion>", self.position)  # bind position method again for cursor tracking

            # bind left mouse button click event to ImageFrame canvas with start_drawing() function
            self.frame.canvas.bind("<Button-1>", start_drawing)
            # bind left mouse button release event to ImageFrame canvas with end_drawing() function
            self.frame.canvas.bind("<ButtonRelease-1>", end_drawing)

            def b_button_press():  # this function will be called when black color button is pressed
                global drawing_color  # make drawing_color global to be able to pass it into outer scope
                # set all buttons relief arguments to RAISED while setting pressed button to SUNKEN
                black_button.config(relief=RAISED)
                red_button.config(relief=RAISED)
                green_button.config(relief=RAISED)
                blue_button.config(relief=RAISED)
                # set pressed button to SUNKEN
                black_button.config(relief=SUNKEN)
                drawing_color = "black"  # set drawing color to black

            def red_button_press():  # this function will be called when red color button is pressed
                global drawing_color  # make drawing_color global to be able to pass it into outer scope
                # set all buttons relief arguments to RAISED while setting pressed button to SUNKEN
                black_button.config(relief=RAISED)
                red_button.config(relief=RAISED)
                green_button.config(relief=RAISED)
                blue_button.config(relief=RAISED)
                # set pressed button to SUNKEN
                red_button.config(relief=SUNKEN)
                drawing_color = "red"  # set drawing color to red

            def green_button_press():  # this function will be called when green color button is pressed
                global drawing_color  # make drawing_color global to be able to pass it into outer scope
                # set all buttons relief arguments to RAISED while setting pressed button to SUNKEN
                black_button.config(relief=RAISED)
                red_button.config(relief=RAISED)
                green_button.config(relief=RAISED)
                blue_button.config(relief=RAISED)
                # set pressed button to SUNKEN
                green_button.config(relief=SUNKEN)
                drawing_color = "green"  # set drawing color to green

            def blue_button_press():  # this function will be called when blue color button is pressed
                global drawing_color  # make drawing_color global to be able to pass it into outer scope
                # set all buttons relief arguments to RAISED while setting pressed button to SUNKEN
                black_button.config(relief=RAISED)
                red_button.config(relief=RAISED)
                green_button.config(relief=RAISED)
                blue_button.config(relief=RAISED)
                # set pressed button to SUNKEN
                blue_button.config(relief=SUNKEN)
                drawing_color = "blue"  # set drawing color to blue

            def finish():  # this function will be called when finish drawing button is pressed
                self.frame.canvas.config(cursor="arrow")  # change cursor icon to default arrow
                self.frame.canvas.unbind("<Button-1>")  # unbind left mouse button click event
                self.frame.canvas.unbind("<ButtonRelease-1>")  # unbind left mouse button release event
                draw_window.destroy()  # destroy drawing adjustment window

            # Master root of frames inside draw_window should be adjust_window.
            frame1 = Frame(draw_window)  # frame1 will contain color buttons
            frame2 = Frame(draw_window)  # frame2 will contain finish drawing button

            # create color buttons with thick borders of 10 pixels to make clear which one is currently pressed
            # default drawing color is black so black button will be initially SUNKEN
            black_button = Button(frame1, bg="#403638", bd=10, relief=SUNKEN, command=b_button_press)
            red_button = Button(frame1, bg="red", bd=10, relief=RAISED, command=red_button_press)
            green_button = Button(frame1, bg="green", bd=10, relief=RAISED, command=green_button_press)
            blue_button = Button(frame1, bg="blue", bd=10, relief=RAISED, command=blue_button_press)

            frame1.pack(side=TOP)  # pack frame1 to the top
            frame2.pack(side=BOTTOM)  # pack frame2 to the bottom

            # pack color buttons in same row, to the LEFT. Give some padding around buttons
            black_button.pack(side=LEFT, ipadx=10, padx=5, pady=10)
            red_button.pack(side=LEFT, ipadx=10, padx=5, pady=10)
            green_button.pack(side=LEFT, ipadx=10, padx=5, pady=10)
            blue_button.pack(side=LEFT, ipadx=10, padx=5, pady=10)

            #  create and pack finish button to frame2. The button will call finish() function
            finish_button = Button(frame2, text="Finish drawing", command=finish)
            finish_button.pack(padx=10, pady=10)
            # If user closes drawing adjustment window through x button in top right corner, apply protocol finish()
            draw_window.protocol("WM_DELETE_WINDOW", finish)


class MainApplication:  # Main application class that contains all other classes and mainloop

    def __init__(self, master, title, geometry):
        """
        Initialize main application window.

        :param master: main root of application
        :param title: title of application window
        :param geometry: sizes of application window
        """
        self.master = master  # main root is master
        self.master.title(title)  # set title of root
        self.master.geometry(geometry)  # adjust geometry of root
        self.main_menu = TopMenu(self.master)  # Create a top menu bar with TopMenu class defined before
        self.master.mainloop()  # start mainloop of application


if __name__ == '__main__':
    root = Tk()  # create main root
    w, h = 1024, 768  # initial sizes of application window
    ws = root.winfo_screenwidth()  # width of computer screen
    hs = root.winfo_screenheight()  # height of computer screen
    x = (ws / 2) - (w / 2)  # initial horizontal position of window
    y = (hs / 2) - (h / 2)  # initial vertical position of window
    root.geometry(f"+{int(x)}+{int(y)}")  # position application window into the middle of computer screen
    icon = PhotoImage(file="picturedit_icon.png")  # create application icon as type PhotoImage
    root.iconphoto(False, icon)  # set application icon with iconphoto() function
    application = MainApplication(root, "Picturedit", f"{w}x{h}")  # create application with given attributes

    # Name: Mert Gulsen
    # University: Istanbul Technical University
