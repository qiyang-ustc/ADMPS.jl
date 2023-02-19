using ADMPS,HDF5

N= 20
matrix = zeros((N,N))
mps = [h5read("./test/data/Iter$(i).h5","Au") for i in 1:N];
for i in 1:N
    for j = 1:N
        matrix[i,j] = ADMPS.overlap(mps[i],mps[j])
    end
end
print(matrix)