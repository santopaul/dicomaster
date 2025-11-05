#!/usr/bin/env python3
"""Generate an anonymized test DICOM file for unit testing."""
import numpy as np
from pydicom.dataset import Dataset, FileDataset

# Create a minimal DICOM dataset
file_meta = Dataset()
file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9.0'
file_meta.ImplementationClassUID = '1.2.3.4.5.6.7.8.9.1'

# Main dataset
ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

# Add required elements
ds.PatientName = 'Anonymous^Test'
ds.PatientID = 'TEST12345'
ds.PatientBirthDate = '20000101'
ds.PatientSex = 'O'
ds.StudyDate = '20250101'
ds.ContentDate = '20250101'
ds.StudyTime = '120000.000000'
ds.ContentTime = '120000.000000'
ds.AccessionNumber = ''
ds.Modality = 'MR'
ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.2'
ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9.3'
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
ds.BodyPartExamined = 'BRAIN'
ds.SeriesDescription = 'Test Series'
ds.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = 'MONOCHROME2'
ds.PixelRepresentation = 0
ds.HighBit = 15
ds.BitsStored = 16
ds.BitsAllocated = 16
ds.Columns = 64
ds.Rows = 64

# Create a small test image (64x64 gradient)
x = np.linspace(0, 1, 64)
y = np.linspace(0, 1, 64)
X, Y = np.meshgrid(x, y)
image = np.uint16(X * Y * 65535)  # Create a 16-bit gradient pattern
ds.PixelData = image.tobytes()

# Save the test file
ds.save_as('tests/fixtures/test.dcm')
