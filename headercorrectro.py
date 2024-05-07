import tkinter as tk
from tkinter import filedialog, simpledialog
from astropy.io import fits


class FITSHeaderEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("FITS Header Editor")

        # Create a select file button
        select_button = tk.Button(root, text="Select FITS File", command=self.select_file)
        select_button.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("FITS Files", "*.fits")])

        if file_path:
            self.file_path = file_path
            self.edit_header()

    def edit_header(self):
        # Open the FITS file and get the 'OBJECT' keyword
        with fits.open(self.file_path, mode='update') as hdu_list:
            object_value = hdu_list[0].header.get('OBJECT', '')

            new_value = simpledialog.askstring("Header Editor", "Current OBJECT field value is: {}. "
                                                                "Please enter new value:".format(object_value))
            if new_value is not None:
                hdu_list[0].header['OBJECT'] = new_value
            else:
                print("No changes made.")

            # Close the fits file
            hdu_list.close()


# Create the Tkinter UI
root = tk.Tk()
app = FITSHeaderEditor(root)
root.mainloop()
