import gc
import importlib

import dearpygui.dearpygui as dpg
import torch
import tree_segmentation
import gui

import extension

model = None
torch.set_anomaly_enabled(True)
while True:
    now = gui.TreeSegmentGUI(model=model)
    now.run()
    if hasattr(now, '_sam'):
        model = now.model
    dpg.stop_dearpygui()
    dpg.destroy_context()
    # dpg.destroy_context()
    del now
    gc.collect()
    # time.sleep(10)
    dpg = importlib.reload(dpg)
    extension = importlib.reload(extension)
    tree_segmentation = importlib.reload(tree_segmentation)
    gui = importlib.reload(gui)
