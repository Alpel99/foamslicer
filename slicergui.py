import tkinter.ttk as ttk
import tkinter as tk
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
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

        self.grid(row=0,column=0,sticky="nsew")
        self.tool_bar = ttk.Frame(parent, width=120)
        self.tool_bar.grid(row=0, column=1)
        self.parent.grid_columnconfigure(1, minsize=200, weight=0)
                
        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.slicer = Foamslicer()
        if self.slicer.config.input_file == "":
            self.getFiles()

        self.setupToolbar()
        self.slicer.readFiles()
        self.setupPlot()
        self.initData()

    def setupToolbar(self):
        tk.Button(self.tool_bar, text="Init Data", command=self.initData).grid(columnspan=3, sticky="nsew")
        tk.Button(self.tool_bar, text="Empty Plot", command=self.resetPlot).grid(columnspan=3, sticky="nsew")

        label = tk.Label(self.tool_bar, text="Axis selection:")
        label.grid(column=0, columnspan=3, sticky="nsew")
        self.axis = tk.IntVar(value=0)
        self.radio1 = tk.Radiobutton(self.tool_bar, text="X", variable=self.axis, value=0).grid(row=2, column=0, sticky="nsew")
        self.radio2 = tk.Radiobutton(self.tool_bar, text="Y", variable=self.axis, value=1).grid(row=2, column=1, sticky="nsew")
        self.radio3 = tk.Radiobutton(self.tool_bar, text="Z", variable=self.axis, value=2).grid(row=2, column=2, sticky="nsew")

        tk.Button(self.tool_bar, text="Rotate Mesh", command=self.rotateMesh).grid(columnspan=3, sticky="nsew")

        tk.Button(self.tool_bar, text="Flip Mesh", command=self.flipMesh).grid(columnspan=3, sticky="nsew")
        
        tk.Button(self.tool_bar, text="Extreme Points", command=self.extremePoints).grid(columnspan=3, sticky="nsew")

        tk.Button(self.tool_bar, text="Curve Padding", command=self.curvePadding).grid(columnspan=3, sticky="nsew")
        # need padding value in gui

        tk.Button(self.tool_bar, text="Even Points", command=self.evenPoints).grid(columnspan=3, sticky="nsew")

        tk.Button(self.tool_bar, text="Extend Points", command=self.extendPoints).grid(columnspan=3, sticky="nsew")

        tk.Button(self.tool_bar, text="Generate GCODE", command=self.generateGcode).grid(columnspan=3, sticky="nsew")
        tk.Button(self.tool_bar, text="Open New", command=self.getFiles).grid(columnspan=3, sticky="nsew")

        # Apply padding to all widgets in the toolbar
        for child in self.tool_bar.winfo_children():
            child.grid_configure(pady=3)

    def generateGcode(self):
        try:
            res = self.slicer.writeGCode()
        except Exception as e:
            print("Error writing GCODE", e)
            res = False
        if(res):
            tk.messagebox.showinfo("GCODE", "GCODE write success")

    def extendPoints(self):
        self.slicer.getExtendedPoints()
        self.slicer.applyShapeOffset()
        self.plot2d(self.slicer.cp1e)
        self.plot2d(self.slicer.cp2e)


    def evenPoints(self):
        self.slicer.getSplines()
        self.slicer.getPointsFromSplines()
        self.plot2d(self.slicer.cp1)
        self.plot2d(self.slicer.cp2)
    
    def curvePadding(self):
        self.slicer.curveNormalization()
        self.plot2d(self.slicer.c1)
        self.plot2d(self.slicer.c2)

    def initData(self):
        self.slicer.getPoints()
        self.plot3d()

    def extremePoints(self):
        self.slicer.shiftMesh()
        self.slicer.config.dim_index = self.axis.get()
        self.slicer.getMeshMaxMin()
        self.slicer.getOrderedExtremePoints()
        self.plot2d(self.slicer.c1)
        self.plot2d(self.slicer.c2)

    def flipMesh(self):
        self.slicer.config.dim_flip_x = self.axis.get() == 0
        self.slicer.config.dim_flip_y = self.axis.get() == 1
        self.slicer.config.dim_flip_z = self.axis.get() == 2
        self.slicer.config.dim_index = None
        self.slicer.flipMesh()
        self.plot3d()

    def rotateMesh(self):
        self.slicer.config.trapz_index = self.axis.get()
        self.slicer.alignMeshAxis()
        self.plot3d()

    def plot3d(self):
        if '2d' in self.ax.name:
            self.resetPlot()
        self.ax.remove()
        self.ax = self.figure.add_subplot(projection='3d')
        points = self.slicer.points
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        self.ax.set_aspect('equal', adjustable='box')
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
            self.slicer.config.input_file = list(files)

    def plot(self, *args):
        plt.plot(args)
        self.figure.canvas.draw()

    def on_closing(self):
        plt.close()
        self.parent.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).grid(row=0,column=0)
    root.mainloop()