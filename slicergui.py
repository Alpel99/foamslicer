import tkinter.ttk as ttk
import tkinter as tk
from tkinter import scrolledtext, messagebox

import matplotlib
import matplotlib.pyplot as plt
from foamslicer import Foamslicer

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        self.style = ttk.Style()
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title('Foamslicer')
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

        self.grid(row=0,column=0,sticky="nsew")
        self.tool_bar = ttk.Frame(parent, width=120)
        self.tool_bar.grid(row=0, column=1)
        self.parent.grid_columnconfigure(1, minsize=200, weight=0)
                
        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.slicer = Foamslicer()
        self.setupPlot()
        if not self.slicer.config.input_file:
            self.getFiles()

        # self.setupToolbar()
        if self.slicer.config.input_file:
            self.slicer.readFiles()
            self.initData()

    def setupToolbar(self):
        for child in self.tool_bar.winfo_children():
            child.destroy()

        tk.Button(self.tool_bar, text="Open New", command=self.getFiles).grid(row=0, columnspan=3, sticky="nsew")
           
        tk.Button(self.tool_bar, text="Init Data", command=self.initData).grid(row=1, columnspan=3, sticky="nsew")
        tk.Button(self.tool_bar, text="Empty Plot", command=self.resetPlot).grid(row=2, columnspan=3, sticky="nsew")


        label = tk.Label(self.tool_bar, text="Axis selection:")
        label.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.axis = tk.IntVar(value=0)
        self.radio1 = tk.Radiobutton(self.tool_bar, text="X", variable=self.axis, value=0).grid(row=4, column=0, sticky="nsew")
        self.radio2 = tk.Radiobutton(self.tool_bar, text="Y", variable=self.axis, value=1).grid(row=4, column=1, sticky="nsew")
        self.radio3 = tk.Radiobutton(self.tool_bar, text="Z", variable=self.axis, value=2).grid(row=4, column=2, sticky="nsew")

        tk.Button(self.tool_bar, text="Rotate Mesh", command=self.rotateMesh).grid(row=5, columnspan=3, sticky="nsew")
        tk.Button(self.tool_bar, text="Align Min", command=self.alignMin).grid(row=3, column=0, sticky="nsew")
        tk.Button(self.tool_bar, text="Align Mid", command=self.alignMid).grid(row=3, column=1, sticky="nsew")
        tk.Button(self.tool_bar, text="Align Max", command=self.alignMax).grid(row=3, column=2, sticky="nsew")

        tk.Button(self.tool_bar, text="Flip Mesh", command=self.flipMesh).grid(row=6, columnspan=3, sticky="nsew")
        
        tk.Button(self.tool_bar, text="Extreme Points", command=self.extremePoints).grid(row=7, columnspan=3, sticky="nsew")

        self.cpad = tk.Button(self.tool_bar, text="Curve Padding", command=self.curvePadding, state="disabled")
        self.cpad.grid(row=8, columnspan=3, sticky="nsew")
        # tk.Label(self.tool_bar, text="Padding Size").grid(row=9, column=0)
        # self.paddingSize = tk.Entry(self.tool_bar)
        # self.paddingSize.grid(row=9, column=1)
        # self.paddingSize.insert(tk.END, self.slicer.config.hotwire_width)

        tk.Label(self.tool_bar, text="Num Points").grid(row=9, column=0)
        self.numPoints = tk.Entry(self.tool_bar)
        self.numPoints.grid(row=9, column=1)
        self.numPoints.insert(tk.END, self.slicer.config.num_points)
        tk.Label(self.tool_bar, text="Num Splines").grid(row=10, column=0)
        self.numSegments = tk.Entry(self.tool_bar)
        self.numSegments.grid(row=10, column=1)
        self.numSegments.insert(tk.END, self.slicer.config.num_segments)
        self.evpoints = tk.Button(self.tool_bar, text="Even Points", command=self.evenPoints, state="disabled")
        self.evpoints.grid(columnspan=3, sticky="nsew")
        
        if(self.slicer.dxf):
            tk.Label(self.tool_bar, text="WorkP Size").grid(row=12, column=0)
            self.workPSize = tk.Entry(self.tool_bar)
            self.workPSize.grid(row=12, column=1)
            self.workPSize.insert(tk.END, self.slicer.config.workpiece_size)
        self.extpoints = tk.Button(self.tool_bar, text="Extend Points", command=self.extendPoints, state="disabled")
        self.extpoints.grid(row=13, column=0, columnspan=2, sticky="nsew")

        self.ext3d = tk.Button(self.tool_bar, text="3D", command=self.switch3dExtended, state=self.extpoints.cget("state"), relief="raised")
        self.ext3d.grid(row=13, column=2,columnspan=1, sticky="nsew")

        self.gengcode = tk.Button(self.tool_bar, text="Generate GCODE", command=self.generateGcode, state="active")
        self.gengcode.grid(row=14, columnspan=3, sticky="nsew")

        self.gengcode = tk.Button(self.tool_bar, text="Open Config", command=self.openConfig)
        self.gengcode.grid(row=15, columnspan=3, sticky="nsew")

        # Apply padding to all widgets in the toolbar
        for child in self.tool_bar.winfo_children():
            child.grid_configure(pady=3, padx=3)

    def openConfig(self):
        self.top = tk.Toplevel(self.parent)
        self.top.title("Config Editor")
        self.top.geometry("900x600")
        
        # Text widget with scrollbar
        self.text_editor = scrolledtext.ScrolledText(self.top, width=40, height=10)
        
        # Pack the text editor and scrollbar
        self.text_editor.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Load the config.json file into the text editor
        self.load_config(self.text_editor)

        self.top.bind('<Control-s>', self.save_config)
        # Save Button
        save_button = tk.Button(self.top, text="Save Config", command=self.save_config)
        save_button.pack(side=tk.RIGHT, padx=10, pady=10)

    def load_config(self, text_editor):
        try:
            with open("config.json", 'r') as f:
                config_content = f.read()
                text_editor.insert('1.0', config_content)
        except FileNotFoundError:
            messagebox.showerror("Error", "Config file not found!")

    def save_config(self):
        try:
            with open("config.json", 'w') as f:
                config_content = self.text_editor.get('1.0', tk.END)
                f.write(config_content)
                # messagebox.showinfo("Success", "Config file saved successfully!")
                self.top.destroy()  # Close the editor window after saving
            self.slicer.config.getConfig()
            self.setupToolbar()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config file: {str(e)}")

    def switch3dExtended(self, swap=True):
        if swap: self.ext3d.config(relief="raised" if self.ext3d.cget("relief") == 'sunken' else "sunken")
        if self.ext3d.cget("relief") == 'raised':
            self.plot2d(self.slicer.cp1e, label="front")
            self.plot2d(self.slicer.cp2e, label="back")
        else:
            self.plot3d(self.slicer.points3d)

    def generateGcode(self):
        try:
            res = self.slicer.writeGCode()
        except Exception as e:
            print("Error writing GCODE", e)
            res = False
        if(res):
            tk.messagebox.showinfo("GCODE", "GCODE write success")

    def extendPoints(self):
        wPSize = int(self.workPSize.get()) if self.workPSize.get() else self.slicer.config.workpiece_size
        self.slicer.config.workpiece_size = wPSize
        self.slicer.getExtendedPoints()
        self.slicer.applyShapeOffset()
        if self.slicer.cp1e is not None and self.slicer.cp2e is not None:
            self.gengcode.config(state="normal")
            self.ext3d.config(state="normal")
        self.slicer.generate3DPoints()
        self.switch3dExtended(False)
        # self.plot2d(self.slicer.cp1e, label="front")
        # self.plot2d(self.slicer.cp2e, label="back")


    def evenPoints(self):
        nsegs = int(self.numSegments.get()) if self.numSegments.get() else self.slicer.config.num_segments
        self.slicer.config.num_segments = nsegs
        npts = int(self.numPoints.get()) if self.numPoints.get() else self.slicer.config.num_points
        self.slicer.config.num_points = npts
        self.slicer.getSplines()
        self.slicer.getPointsFromSplines()
        if self.slicer.cp1 is not None and self.slicer.cp2 is not None:
            self.extpoints.config(state="normal")
        self.plot2d(self.slicer.cp1, label="front")
        self.plot2d(self.slicer.cp2, label="back")
    
    def curvePadding(self):
        # hww = float(self.paddingSize.get()) if self.paddingSize.get() else self.slicer.config.hotwire_width
        # self.slicer.config.hotwire_width = hww
        self.slicer.curveNormalization()
        self.plot2d(self.slicer.c1, label="front")
        self.plot2d(self.slicer.c2, label="back")

    def initData(self):
        self.setupToolbar()
        if self.slicer.dxf:
            self.plot2d(self.slicer.c1, label="front")
            self.plot2d(self.slicer.c2, label="back")
            self.evpoints.config(state="normal")
            self.cpad.config(state="normal")
        else:
            self.slicer.getPoints()
            self.plot3d(self.slicer.points)

    def extremePoints(self):
        self.slicer.shiftMesh()
        self.slicer.config.dim_index = self.axis.get()
        self.slicer.getMeshMaxMin()
        self.slicer.getOrderedExtremePoints()
        if self.slicer.c1 is not None and self.slicer.c2 is not None:
            self.evpoints.config(state="normal")
            self.cpad.config(state="normal")
        self.plot2d(self.slicer.c1, label="front")
        self.plot2d(self.slicer.c2, label="back")

    def flipMesh(self):
        self.slicer.config.dim_flip_x = self.axis.get() == 0
        self.slicer.config.dim_flip_y = self.axis.get() == 1
        self.slicer.config.dim_flip_z = self.axis.get() == 2
        self.slicer.config.dim_index = None
        self.slicer.flipMesh()
        if self.slicer.dxf:
            self.plot2d(self.slicer.c1, label="front")
            self.plot2d(self.slicer.c2, label="back")
        else:
            self.plot3d(self.slicer.points)

    def alignMin(self):
        self.slicer.config.dim_index = self.axis.get()
        self.slicer.config.mode = 0
        self.slicer.alignMesh()
        self.plot3d(self.slicer.points)

    def alignMax(self):
        self.slicer.config.dim_index = self.axis.get()
        self.slicer.config.mode = 1
        self.slicer.alignMesh()
        self.plot3d(self.slicer.points)

    def alignMid(self):
        self.slicer.config.dim_index = self.axis.get()
        self.slicer.config.mode = 2
        self.slicer.alignMesh()
        self.plot3d(self.slicer.points)

    def rotateMesh(self):
        self.slicer.config.trapz_idx = self.axis.get()
        self.slicer.alignMeshAxis()
        self.plot3d(self.slicer.points)

    def plot3d(self, points):
        if not self.slicer.dxf:
            self.slicer.shiftMesh()
        if '2d' in self.ax.name:
            self.resetPlot()
        self.ax.remove()
        self.ax = self.figure.add_subplot(projection='3d')
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # self.ax.view_init(0,0)
        self.figure.canvas.draw()

    def resetPlot(self):
        self.ax.remove()
        self.ax = self.figure.add_subplot()
        self.figure.canvas.draw()

    def plot2d(self, points, label=None):
        if '3d' in self.ax.name:
            self.resetPlot()
        self.ax.plot(points[:, 0], points[:, 1], 'o-', label=label)
        self.ax.set_aspect('equal')
        # self.ax.legend()
        self.legend = self.ax.legend(loc='lower center', bbox_to_anchor=(0.5,1))
        self.ax.add_artist(self.legend)
        self.figure.canvas.draw()


    def setupPlot(self):
        self.figure = plt.figure(figsize=(6,4), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.toolbar_frame = tk.Frame(self.parent)
        self.toolbar_frame.grid(row=1,column=0)
        NavigationToolbar2Tk(self.figure_canvas, self.toolbar_frame)
        self.ax = self.figure.add_subplot()
        # self.ax.plot([1,2,3,4],[7,3,2,1],label="test")
        self.figure_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def getFiles(self):
            files = tk.filedialog.askopenfilenames(parent=self.parent, title='Choose files (stl or dxf)')
            if files is not None:
                self.slicer.config.input_file = list(files)
            else:
                raise("No file selected")
            self.slicer.readFiles()
            self.initData()

    def plot(self, *args):
        plt.plot(args)
        self.figure.canvas.draw()

    def on_closing(self):
        plt.close()
        self.slicer.config.writeConfig()
        self.parent.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).grid(row=0,column=0)
    root.mainloop()