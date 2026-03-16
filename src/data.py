import zarr

z = zarr.open("AR0047-C.gatk.zarr", mode="r")

gt = z["AR0047-C"]["2L"]["calldata"]["GT"]

print(gt.shape)

chunk = gt[:10000, 0, :]

print(chunk.shape)

encoded = chunk.sum(axis=1)

print(encoded[:20])