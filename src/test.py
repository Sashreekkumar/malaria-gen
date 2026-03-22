import zarr
store = zarr.open("/home/sashreekkumar/Documents/Projects/malariagen/extracted/2116/gt", mode='r')
print(store)
print(store.shape)