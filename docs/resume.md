# Segmented encoding and resume

Set `segment_frames` to divide a mosaic encode into fixed-size MP4 parts. Each
part is written under `output/segments/part_0000.mp4`, followed by a checkpoint
that records the part's SHA-256, its frame count, the run identity, and the
matcher state after its last frame. A short final part records the actual EOF;
an exact multiple of the segment size does not add an empty part.

If a process stops, run the same command again with the same source, gallery,
and video settings. The encoder validates each committed part and restores the NumPy
random generator and held tile choices from its paired checkpoint. It then
seeks to the next source frame and encodes the remaining parts. A source,
gallery, or video setting change produces a clear mismatch error. A missing or
altered completed part is reported as corruption instead of being combined.

The run identity includes the resolved source path, size and modification time,
the gallery fingerprint, video metadata, output-affecting configuration, and a
checkpoint code version. Keep the segment directory with the output to retain
resume capability. Concatenation uses ffmpeg stream copy after every part is
complete.

All runs with segmentation enabled use the same fixed boundaries, so a resumed
run decodes to the same frames as an uninterrupted run with the same settings.
The independently encoded parts can have different bytes from one unsegmented
encode.
