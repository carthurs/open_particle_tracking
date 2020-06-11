from __future__ import print_function
import vtk

if __name__ == "__main__":
	domain_particles = vtk.vtkXMLUnstructuredGridReader()
	domain_particles.SetFileName("particles.vtu")
	domain_particles.Update()

	particles_data = domain_particles.GetOutput()
	print("Number of nodes in the input mesh:", particles_data.GetNumberOfPoints())

	particles_data_to_polydata_filter = vtk.vtkGeometryFilter()
	particles_data_to_polydata_filter.SetInput(domain_particles.GetOutput())
	particles_data_to_polydata_filter.Update()

	particles_data_polydata_format = particles_data_to_polydata_filter.GetOutput()


	mesh_decimator = vtk.vtkDecimatePro()
	mesh_decimator.SetInput(particles_data_to_polydata_filter.GetOutput())

	mesh_decimator.SetTargetReduction(0.8) # 90% reduction (10% remaining)
	mesh_decimator.Update()

	print("Number of nodes in the output mesh:", mesh_decimator.GetOutput().GetNumberOfPoints())

	polydata_to_unstructured_grid_converter = vtk.vtkAppendFilter()
	polydata_to_unstructured_grid_converter.AddInput(mesh_decimator.GetOutput())
	polydata_to_unstructured_grid_converter.Update()


	writer = vtk.vtkXMLUnstructuredGridWriter()
	writer.SetFileName("decimated_particles.vtu")
	if vtk.VTK_MAJOR_VERSION == 6:
	    writer.SetInputData(polydata_to_unstructured_grid_converter)
	else:
	    writer.SetInput(polydata_to_unstructured_grid_converter.GetOutput())
	writer.Update()