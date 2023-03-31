from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.unmixing.unmixer import SpectralUnmixer, SO2Calculator

def linear_unmixing(spectra, wavelengths):

    r = Reconstruction(spectra, wavelengths,
                       field_of_view=(1, 1, 1))  # field of view is the width of the image along x, y, z
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_X"] = 1
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Y"] = 1
    r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Z"] = 1
    um = SpectralUnmixer(["Hb", "HbO2"], r.wavelengths)
    so = SO2Calculator()
    um, _, _ = um.run(r, None)
    so2, _, _ = so.run(um, None)

    return so2