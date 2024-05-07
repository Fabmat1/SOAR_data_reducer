import json
import os
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.ttk as ttk
from data_reduction import data_reduction
from PIL import ImageTk, Image


class DataReductionGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        if os.path.isfile("./.previousselection.json"):
            with open("./.previousselection.json", "r") as jsonfile:
                self.variabledict = json.load(jsonfile)
        else:
            self.variabledict = {
                "unshiftedflats": [],
                "shiftedflats": [],
                "biases": [],
                "scienceframes": [],
                "comparisonframes": [],
                "compdivision": 3,
                "sciencedivision": 3,
                "output_dict": "./output",
                "outputfile_path": "./SOAR.csv",
                "comparisonparams": [],
            }

        super().__init__()
        self.title("SOAR Data Reducer")
        try:
            if os.name == "nt":
                self.iconbitmap("favicon.ico")
            else:
                imgicon = ImageTk.PhotoImage(Image.open("favicon.ico"))
                self.tk.call('wm', 'iconphoto', self._w, imgicon)
        except Exception as e:
            print(e)
        self.geometry("800x600+0+0")
        if os.name == 'nt':
            self.state('zoomed')
        elif os.name == "posix":
            self.attributes('-zoomed', True)

        mainframe = tk.Frame(self)
        # Shifted, not shifted
        flatframe = tk.Frame(mainframe)
        unshiftedflatframe = tk.Frame(flatframe)
        shiftedflatframe = tk.Frame(flatframe)
        unshiftedlabel = tk.Label(unshiftedflatframe, text="Select the flatframes that were not offset:")
        unshiftedlabel.pack(anchor="w")
        unshiftedselectedfilelabel = tk.Label(unshiftedflatframe, text="", fg="green")
        unshiftedbtn = tk.Button(unshiftedflatframe, text="Select Files", command=lambda: self.fileselection(unshiftedflatframe, unshiftedselectedfilelabel, "unshiftedflats"))
        unshiftedbtn.pack()
        unshiftedselectedfilelabel.pack()
        shiftedlabel = tk.Label(shiftedflatframe, text="Select the flatframes that were offset:")
        shiftedlabel.pack(anchor="w")
        shiftedselectedfilelabel = tk.Label(shiftedflatframe, text="", fg="green")
        shiftedbtn = tk.Button(shiftedflatframe, text="Select Files", command=lambda: self.fileselection(shiftedflatframe, shiftedselectedfilelabel, "shiftedflats"))
        shiftedbtn.pack()
        shiftedselectedfilelabel.pack()

        unshiftedflatframe.pack(anchor="w", padx=10, pady=10)
        shiftedflatframe.pack(anchor="w", padx=10, pady=10)
        flatframe.grid(row=0, column=0)
        biasframe = tk.Frame(mainframe)
        biaslabel = tk.Label(biasframe, text="Select the bias frames:")
        biaslabel.pack(anchor="w")
        biasselectedfilelabel = tk.Label(biasframe, text="", fg="green")
        biasbtn = tk.Button(biasframe, text="Select Files", command=lambda: self.fileselection(biasframe, biasselectedfilelabel, "biases"))
        biasbtn.pack()
        biasselectedfilelabel.pack()
        biasframe.grid(row=1, column=0, sticky="w", padx=10, pady=10)
        scienceframe = tk.Frame(mainframe)
        sciencelabel = tk.Label(scienceframe, text="Select the science frames:")
        sciencelabel.pack(anchor="w")
        scienceselectedfilelabel = tk.Label(scienceframe, text="", fg="green")
        sciencebtn = tk.Button(scienceframe, text="Select Files", command=lambda: self.fileselection(scienceframe, scienceselectedfilelabel, "scienceframes"))
        sciencebtn.pack()
        scienceselectedfilelabel.pack()
        complabel = tk.Label(scienceframe, text="Select the comparison frames:")
        complabel.pack(anchor="w")
        compselectedfilelabel = tk.Label(scienceframe, text="", fg="green")
        compbtn = tk.Button(scienceframe, text="Select Files", command=lambda: self.fileselection(scienceframe, compselectedfilelabel, "comparisonframes"))
        compbtn.pack()
        compselectedfilelabel.pack()
        scienceframe.grid(row=0, column=1, sticky="w", padx=10, pady=10)
        controlframe = tk.Frame(mainframe)
        outputpathframe = tk.Frame(controlframe)
        outputlabel = tk.Label(outputpathframe, text="Output path")
        output_path_var = tk.StringVar(value=self.variabledict["output_dict"])
        output_path_var.trace_add("write", lambda a, b, c: self.set_entry("output_dict", output_path_var.get()))
        output_path = tk.Entry(outputpathframe, validate="focusout", textvariable=output_path_var)
        outputlabel.pack(side=tk.LEFT)
        output_path.pack(side=tk.RIGHT)
        outputpathframe.pack()

        outputfileframe = tk.Frame(controlframe)
        outputfilelabel = tk.Label(outputfileframe, text="Output File")
        output_file_var = tk.StringVar(value=self.variabledict["outputfile_path"])
        output_file_var.trace_add("write", lambda a, b, c: self.set_entry("outputfile_path", output_file_var.get()))
        output_file = tk.Entry(outputfileframe, validate="focusout", textvariable=output_file_var)
        outputfilelabel.pack(side=tk.LEFT)
        output_file.pack(side=tk.RIGHT)
        outputfileframe.pack()

        divframe = tk.Frame(controlframe)
        compdivlabel = tk.Label(divframe, text="Size of Comp Lamp chunks")
        compdivvar = tk.StringVar(value=self.variabledict["compdivision"])
        compdiventry = tk.Entry(divframe, validate="focusout", textvariable=compdivvar)
        compdivvar.trace_add("write", lambda a, b, c: self.set_entry("compdivision", compdivvar.get()))
        compdivlabel.grid(row=0, column=0)
        compdiventry.grid(row=0, column=1)
        sciencedivlabel = tk.Label(divframe, text="Size of Science Frame chunks")
        sciencedivvar = tk.StringVar(value=self.variabledict["sciencedivision"])
        sciencediventry = tk.Entry(divframe, validate="focusout", textvariable=sciencedivvar)
        sciencedivvar.trace_add("write", lambda a, b, c: self.set_entry("sciencedivision", sciencedivvar.get()))
        sciencedivlabel.grid(row=1, column=0)
        sciencediventry.grid(row=1, column=1)
        coaddvar = tk.IntVar(value=0)
        coaddcheck = tk.Checkbutton(divframe, text="Coadd Science Chunks", variable=coaddvar)
        coaddcheck.grid(row=2, column=0, columnspan=1)
        divframe.pack()

        testbtn = tk.Button(controlframe, text="Reduce Data", command=lambda:
        data_reduction(
            self.variabledict["unshiftedflats"],
            self.variabledict["shiftedflats"],
            self.variabledict["biases"],
            self.variabledict["scienceframes"],
            self.variabledict["comparisonframes"],
            self.variabledict["outputfile_path"],
            self.variabledict["output_dict"],
            int(self.variabledict["compdivision"]),
            int(self.variabledict["sciencedivision"]),
            coadd_chunk=True if coaddvar.get() == 1 else False
        ))
        testbtn.pack()
        controlframe.grid(row=1, column=1)

        # ttk.Separator(mainframe, orient=tk.VERTICAL).grid(column=1, row=0, rowspan=2, sticky='ns')
        # ttk.Separator(mainframe, orient=tk.HORIZONTAL).grid(column=0, row=0, columnspan=2, sticky='ew')

        mainframe.pack(fill=tk.BOTH, expand=1)

    def printstuff(self):
        print(f"""unshiftedflats:{self.variabledict["unshiftedflats"]}
shiftedflats:{self.variabledict["shiftedflats"]}
biases:{self.variabledict["biases"]}
scienceframes:{self.variabledict["scienceframes"]}
comparisonframes:{self.variabledict["comparisonframes"]}
comparisonparams:{self.variabledict["comparisonparams"]}""")

    def fileselection(self, wintitle, label, key):
        storevar = fd.askopenfilenames(parent=self, title=wintitle)
        self.variabledict[key] = storevar
        with open(".previousselection.json", "w") as jsonfile:
            json.dump(self.variabledict, jsonfile)
        label.config(text=f"Found {len(storevar)} Files!")

    def set_entry(self, key, value):
        self.variabledict[key] = value
        with open(".previousselection.json", "w") as jsonfile:
            json.dump(self.variabledict, jsonfile)


if __name__ == "__main__":
    root = DataReductionGUI()
    root.mainloop()
