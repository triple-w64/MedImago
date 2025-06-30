#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EchoSRNet Graphical User Interface
Provides GUI to select input images, pre-trained models and save path
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import logging
from pathlib import Path
import torch
from threading import Thread

# Automatically set Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Define TextHandler class
class TextHandler(logging.Handler):
    """Handler for outputting logs to a Tkinter Text widget"""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        
        def append():
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)  # Auto-scroll to the bottom
            self.text_widget.config(state=tk.DISABLED)
            
        # Update UI in a thread-safe way
        self.text_widget.after(0, append)

# Import processing functions from infer.py
def get_infer_functions():
    try:
        # Import related functions
        from infer import list_pretrained_models, load_pytorch_model, process_image
        return list_pretrained_models, load_pytorch_model, process_image
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure infer.py is in the same directory and contains the required functions")
        sys.exit(1)

class EchoSRNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoSRNet Image Super-Resolution")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 设置默认字体大小
        self.font_normal = ("Arial", 25)
        self.font_large = ("Arial", 25)
        self.font_title = ("Arial", 30, "italic")
        
        # Import functions
        self.list_pretrained_models, self.load_pytorch_model, self.process_image = get_infer_functions()
        
        # Get pre-trained model list
        try:
            self.pretrained_models = self.list_pretrained_models()
        except Exception as e:
            self.pretrained_models = []
            print(f"Failed to get pre-trained model list: {e}")
        
        # Store user selections
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.scale_var = tk.IntVar(value=2)
        self.ultrasound_var = tk.BooleanVar(value=False)
        self.rgb_var = tk.BooleanVar(value=False)
        self.cpu_var = tk.BooleanVar(value=False)
        self.show_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        
        # IPG相关选项
        self.use_ipg_var = tk.BooleanVar(value=True)
        self.max_degree_var = tk.IntVar(value=8)
        self.base_patch_size_var = tk.IntVar(value=64)
        
        # Create UI components
        self.create_widgets()
        
        # Setup logger
        self.setup_logger()
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.cpu_var.get() else 'cpu')
        self.logger.info(f"Initialization complete, using device: {self.device}")
        self.logger.info(f"Found {len(self.pretrained_models)} pre-trained models")
    
    def create_widgets(self):
        """Create GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 设置ttk样式
        style = ttk.Style()
        style.configure("TLabel", font=self.font_normal)
        style.configure("TButton", font=self.font_normal)
        style.configure("TCheckbutton", font=self.font_normal)
        style.configure("TRadiobutton", font=self.font_normal)
        style.configure("TEntry", font=self.font_normal)
        
        # Input image
        ttk.Label(main_frame, text="Input Image:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=60, font=self.font_normal).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output path
        ttk.Label(main_frame, text="Output Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(main_frame, textvariable=self.output_path_var, width=60, font=self.font_normal).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Pre-trained model list
        ttk.Label(main_frame, text="Pre-trained Model:").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.model_listbox = tk.Listbox(model_frame, height=6, width=60, font=self.font_normal)
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_scrollbar = ttk.Scrollbar(model_frame, orient="vertical", command=self.model_listbox.yview)
        model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=model_scrollbar.set)
        
        # Fill model list
        if self.pretrained_models:
            for model in self.pretrained_models:
                self.model_listbox.insert(tk.END, model)
        else:
            self.model_listbox.insert(tk.END, "No pre-trained models found")
        
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        # Scale factor
        ttk.Label(main_frame, text="Scale Factor:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        scale_frame = ttk.Frame(main_frame)
        scale_frame.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(scale_frame, text="2x", variable=self.scale_var, value=2).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(scale_frame, text="4x", variable=self.scale_var, value=4).pack(side=tk.LEFT, padx=10)
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options")
        options_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        ttk.Checkbutton(options_frame, text="Ultrasound Mode", variable=self.ultrasound_var).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(options_frame, text="RGB Mode", variable=self.rgb_var).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(options_frame, text="CPU Mode", variable=self.cpu_var).grid(row=0, column=2, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(options_frame, text="Show Result", variable=self.show_var).grid(row=0, column=3, padx=10, pady=5, sticky="w")
        
        # IPG Options
        ipg_frame = ttk.LabelFrame(main_frame, text="IPG Settings")
        ipg_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        ttk.Checkbutton(ipg_frame, text="Enable IPG", variable=self.use_ipg_var).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        ttk.Label(ipg_frame, text="Max Degree:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        max_degree_frame = ttk.Frame(ipg_frame)
        max_degree_frame.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(max_degree_frame, text="4", variable=self.max_degree_var, value=4).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(max_degree_frame, text="8", variable=self.max_degree_var, value=8).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(max_degree_frame, text="12", variable=self.max_degree_var, value=12).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ipg_frame, text="Base Patch Size:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        patch_size_frame = ttk.Frame(ipg_frame)
        patch_size_frame.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(patch_size_frame, text="32", variable=self.base_patch_size_var, value=32).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(patch_size_frame, text="64", variable=self.base_patch_size_var, value=64).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(patch_size_frame, text="128", variable=self.base_patch_size_var, value=128).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        ttk.Label(main_frame, textvariable=self.status_var, font=self.font_title).grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5)
        
        # Process button
        self.process_button = ttk.Button(button_frame, text="Start Processing", command=self.process_image_wrapper)
        self.process_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Process with all models button
        self.process_all_button = ttk.Button(button_frame, text="Process with All Models", command=self.process_with_all_models)
        self.process_all_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Create log area
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        self.log_text = tk.Text(log_frame, height=10, width=80, font=self.font_normal)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Set grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
    
    def setup_logger(self):
        """Set up logging handler"""
        # Use Text widget as log handler
        self.text_handler = TextHandler(self.log_text)
        self.logger = logging.getLogger('infer_gui')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Add handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.text_handler.setFormatter(formatter)
        self.logger.addHandler(self.text_handler)
    
    def browse_input(self):
        """Select input image"""
        filetypes = [
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"),
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg *.jpeg"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        if file_path:
            self.input_path_var.set(file_path)
            
            # Automatically set output path
            if not self.output_path_var.get():
                input_file = Path(file_path)
                output_file = str(input_file.parent / f"{input_file.stem}_SR{input_file.suffix}")
                self.output_path_var.set(output_file)
    
    def browse_output(self):
        """Select output path"""
        filetypes = [
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.asksaveasfilename(
            title="Save Output Image",
            filetypes=filetypes,
            defaultextension=".png"
        )
        if file_path:
            self.output_path_var.set(file_path)
    
    def on_model_select(self, event):
        """Triggered when model is selected from list"""
        if not self.pretrained_models:
            return
            
        selection = self.model_listbox.curselection()
        if selection:
            index = selection[0]
            model_name = self.model_listbox.get(index)
            self.model_var.set(model_name)
            
            # If ultrasound model, enable ultrasound mode automatically
            if 'us' in model_name.lower():
                self.ultrasound_var.set(True)
                self.logger.info(f"Detected ultrasound model {model_name}, enabled ultrasound mode automatically")
            
            # Infer scale factor from model name
            if 'scale4' in model_name.lower():
                self.scale_var.set(4)
                self.logger.info("Set scale factor to 4x")
            else:
                self.scale_var.set(2)
                self.logger.info("Set scale factor to 2x")
                
            # Handle different model types
            if 'ipggnnechospannet' in model_name.lower() or 'ipg' in model_name.lower():
                self.logger.info(f"Detected IPGGNNEchoSPANNet model {model_name}")
                # 自动启用IPG功能
                self.use_ipg_var.set(True)
                self.logger.info("Automatically enabled IPG features for IPGGNNEchoSPANNet model")
                # IPGGNNEchoSPANNet parameters are set in _create_args
            elif 'echospannet' in model_name.lower():
                self.logger.info(f"Detected EchoSPANNet model {model_name}")
                # 对于普通EchoSPANNet模型，禁用IPG功能
                self.use_ipg_var.set(False)
                # EchoSPANNet parameters are set in _process_image_thread
            elif 'fsrcnn' in model_name.lower():
                self.logger.info(f"Detected FSRCNN model {model_name}")
                self.use_ipg_var.set(False)
            elif 'srcnn' in model_name.lower():
                self.logger.info(f"Detected SRCNN model {model_name}")
                self.use_ipg_var.set(False)
            elif 'lapsrn' in model_name.lower():
                self.logger.info(f"Detected LapSRN model {model_name}")
                self.use_ipg_var.set(False)
            elif 'edsr' in model_name.lower():
                self.logger.info(f"Detected EDSR model {model_name}")
                self.use_ipg_var.set(False)
            elif 'vdsr' in model_name.lower():
                self.logger.info(f"Detected VDSR model {model_name}")
                self.use_ipg_var.set(False)
            elif 'echosrnet' in model_name.lower():
                self.logger.info(f"Detected EchoSRNet model {model_name}")
                self.use_ipg_var.set(False)
    
    def validate_inputs(self):
        """Validate inputs are valid"""
        if not self.input_path_var.get():
            messagebox.showerror("Error", "Please select an input image")
            return False
            
        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an output path")
            return False
            
        # 当使用所有模型时不需要单独选择模型
        if not self.model_var.get() and not hasattr(self, 'process_all_models_flag'):
            messagebox.showerror("Error", "Please select a pre-trained model")
            return False
            
        # Check if files exist
        if not os.path.exists(self.input_path_var.get()):
            messagebox.showerror("Error", f"Input file does not exist: {self.input_path_var.get()}")
            return False
            
        # 使用所有模型时不检查单个模型文件
        if not hasattr(self, 'process_all_models_flag') and self.model_var.get():
            model_path = os.path.join(current_dir, 'trained', self.model_var.get())
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file does not exist: {model_path}")
                return False
            
        return True
    
    def process_image_wrapper(self):
        """Wrapper function for processing image, used to start a new thread"""
        if not self.validate_inputs():
            return
            
        # Disable process button
        self.process_button.config(state=tk.DISABLED)
        self.process_all_button.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        
        # Run processing in a new thread to avoid freezing GUI
        thread = Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()

    def process_with_all_models(self):
        """使用所有预训练模型处理图像"""
        if not self.pretrained_models:
            messagebox.showerror("Error", "No pre-trained models available")
            return
            
        # 标记为使用所有模型模式
        self.process_all_models_flag = True
        
        if not self.validate_inputs():
            delattr(self, 'process_all_models_flag')
            return
            
        # Disable process buttons
        self.process_button.config(state=tk.DISABLED)
        self.process_all_button.config(state=tk.DISABLED)
        self.status_var.set("Processing with all models...")
        
        # Run processing in a new thread
        thread = Thread(target=self._process_all_models_thread)
        thread.daemon = True
        thread.start()
    
    def _process_all_models_thread(self):
        """在新线程中使用所有模型处理图像"""
        try:
            # 获取输入和输出路径
            input_path = self.input_path_var.get()
            base_output_path = self.output_path_var.get()
            
            # 创建输出目录
            output_dir = os.path.dirname(base_output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取输入文件基本信息
            input_file = Path(input_path)
            input_stem = input_file.stem
            input_suffix = input_file.suffix
            
            # 处理每个模型
            total_models = len(self.pretrained_models)
            success_count = 0
            
            for i, model_name in enumerate(self.pretrained_models):
                self.status_var.set(f"Processing model {i+1}/{total_models}: {model_name}")
                self.logger.info(f"Processing with model: {model_name}")
                
                try:
                    # 设置当前模型
                    self.model_var.set(model_name)
                    
                    # 根据模型名称设置参数
                    if 'us' in model_name.lower():
                        self.ultrasound_var.set(True)
                    else:
                        self.ultrasound_var.set(False)
                        
                    if 'scale4' in model_name.lower():
                        self.scale_var.set(4)
                    else:
                        self.scale_var.set(2)
                    
                    # 设置输出文件名（包含模型名称）
                    model_suffix = model_name.replace('.pth', '')
                    output_file = os.path.join(output_dir, f"{input_stem}_{model_suffix}{input_suffix}")
                    
                    # 设置参数
                    args = self._create_args()
                    args.output = output_file
                    
                    # 加载模型
                    model_path = os.path.join(current_dir, 'trained', model_name)
                    if not os.path.exists(model_path):
                        self.logger.error(f"Model file not found: {model_path}")
                        continue
                        
                    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
                    model = self.load_pytorch_model(args, device, self.logger)
                    
                    # 处理图像
                    output_img, inference_time, metrics = self.process_image(
                        input_path, model, device, args, self.logger, None
                    )
                    
                    if output_img:
                        # 保存结果
                        output_img.save(output_file)
                        self.logger.info(f"Result saved to: {output_file}")
                        
                        # 显示最后一个结果
                        if args.show and i == total_models - 1:
                            output_img.show()
                            
                        success_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing model {model_name}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            self.status_var.set(f"Completed {success_count}/{total_models} models")
            self.logger.info(f"All models processing complete. Success: {success_count}/{total_models}")
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            self.status_var.set(f"Processing error")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # 移除标记
            if hasattr(self, 'process_all_models_flag'):
                delattr(self, 'process_all_models_flag')
                
            # Re-enable process buttons
            self.process_button.config(state=tk.NORMAL)
            self.process_all_button.config(state=tk.NORMAL)
    
    def _create_args(self):
        """创建参数对象"""
        class Args:
            pass
                
        args = Args()
        args.model = os.path.join(current_dir, 'trained', self.model_var.get())
        args.input = self.input_path_var.get()
        args.output = self.output_path_var.get()
        args.scale = self.scale_var.get()
        args.ultrasound_mode = self.ultrasound_var.get()
        args.rgb_mode = self.rgb_var.get()
        args.cpu = self.cpu_var.get()
        args.show = self.show_var.get()
        args.no_save = False
        args.verbose = True
        args.m = 5
        args.int_features = 16
        args.feature_size = 256
        args.onnx_model = None
        args.use_speckle_filter = None
        args.use_edge_enhancer = None
        args.use_signal_enhancer = None
        args.reference = None
        
        # EchoSPANNet特有参数
        args.span_feature_channels = 48
        args.span_num_blocks = 6
        args.use_fourier = True
        
        # IPGGNNEchoSPANNet特有参数
        args.use_ipg = self.use_ipg_var.get()
        args.max_degree = self.max_degree_var.get()
        args.base_patch_size = self.base_patch_size_var.get()
        
        return args
    
    def _process_image_thread(self):
        """Process image in a new thread"""
        try:
            # 创建参数
            args = self._create_args()
            
            # 设置设备
            if args.cpu:
                device = torch.device('cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {device}")
            
            # 加载模型
            self.logger.info(f"Loading model: {self.model_var.get()}")
            model = self.load_pytorch_model(args, device, self.logger)
            
            # 处理图像
            self.logger.info(f"Processing image: {args.input}")
            input_path = Path(args.input)
            output_img, inference_time, metrics = self.process_image(
                str(input_path), model, device, args, self.logger, None
            )
            
            if output_img:
                # Save result
                # Check if output is a complete file path (with extension) or a directory
                if os.path.splitext(args.output)[1]:  # Has extension, treat as file
                    output_file = args.output
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                else:  # No extension, treat as directory
                    # Ensure output directory exists
                    os.makedirs(args.output, exist_ok=True)
                    # Build output filename from input filename
                    output_file = os.path.join(args.output, f"{input_path.stem}_x{args.scale}{input_path.suffix}")
                
                output_img.save(output_file)
                self.logger.info(f"Result saved to: {output_file}")
                
                # Show result
                if args.show:
                    output_img.show()
                
                self.status_var.set(f"Processing complete (inference time: {inference_time:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            self.status_var.set(f"Processing error")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Re-enable process button
            self.process_button.config(state=tk.NORMAL)
            self.process_all_button.config(state=tk.NORMAL)

def main():
    try:
        root = tk.Tk()
        app = EchoSRNetGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        
        # If error, print usage instructions
        print("\nUsage Instructions:")
        print("1. Ensure tkinter library is installed and working properly")
        print("2. Ensure infer.py is in the same directory")
        print("3. Ensure there are pre-trained models in the trained folder")
        print("4. If running in WSL or remote environment, configure X11 forwarding")

if __name__ == "__main__":
    main() 