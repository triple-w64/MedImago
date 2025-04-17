# vtk_dcm_recon.py
import vtk
import os

class VTKReconTab:
    def __init__(self, dicom_directory, threshold, output_filename):
        self.dicom_directory = dicom_directory
        self.threshold = threshold
        self.output_filename = output_filename

    def load_dicom_series(self):
        dicom_reader = vtk.vtkDICOMImageReader()
        dicom_reader.SetDirectoryName(self.dicom_directory)
        dicom_reader.Update()
        return dicom_reader

    def apply_gaussian_smoothing(self, input_port):
        smoother = vtk.vtkImageGaussianSmooth()
        smoother.SetDimensionality(3)
        smoother.SetStandardDeviation(1.5, 1.5, 1.5)
        smoother.SetInputConnection(input_port)
        smoother.Update()
        return smoother

    def perform_reconstruction(self):
        dicom_reader = self.load_dicom_series()
        smoother = self.apply_gaussian_smoothing(dicom_reader.GetOutputPort())

        # Setup volume rendering
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputConnection(smoother.GetOutputPort())

        color_function = vtk.vtkColorTransferFunction()
        color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_function.AddRGBPoint(255, 1.0, 1.0, 1.0)

        opacity_function = vtk.vtkPiecewiseFunction()
        opacity_function.AddPoint(self.threshold, 0.0)
        opacity_function.AddPoint(255, 1.0)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function)
        volume_property.SetInterpolationTypeToLinear()

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer = vtk.vtkRenderer()
        renderer.AddVolume(volume)
        renderer.SetBackground(0.1, 0.2, 0.4)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        render_window.Render()
        interactor.Start()

    def export_to_stl(self):
        dicom_reader = self.load_dicom_series()
        smoother = self.apply_gaussian_smoothing(dicom_reader.GetOutputPort())

        contour_filter = vtk.vtkContourFilter()
        contour_filter.SetValue(0, self.threshold)
        contour_filter.SetInputConnection(smoother.GetOutputPort())
        contour_filter.Update()

        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetInputConnection(contour_filter.GetOutputPort())
        stl_writer.SetFileName(self.output_filename)
        stl_writer.Write()
