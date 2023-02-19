using ADMPS
using ADMPS: num_grad,Zofβ,logoverlap,Z,obs_env,magofβ,eneofβ,overlap,onestep,isingβc,init_mps
using CUDA
using KrylovKit
using LinearAlgebra: svd, norm
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote

# @testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
#     Random.seed!(100)
#     D,χ = 2,5
#     β = 0.4
#     model = Ising(β)
#     M = atype{dtype}(model_tensor(model))
#     mps = init_mps(D = D, χ = χ)
#     Au = mps
#     Ad = mps
#     ff(Ad) = logoverlap(Au, Ad, M)
#     # @show logoverlap(Au, mps, M),Zygote.gradient(ff,Ad)
#     gradzygote = first(Zygote.gradient(mps) do x
#         logoverlap(Au, x, M)
#     end)
#     gradnum = num_grad(mps, δ=1e-4) do x
#         logoverlap(Au, x, M)
#     end
#     @test gradzygote ≈ gradnum atol=1e-5
# end

# @testset "onestep optimize Ad with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
#     seed_number = 100
#     β = 0.8
#     χ = 10
#     maxiter = 5
#     infolder, outfolder = "./data/", "./data/"
#     model = Ising(β)
#     M = atype{dtype}(model_tensor(model))
#     Random.seed!(seed_number)

#     Ad = onestep(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", χ = χ, verbose= true, savefile = false)
#     @test Ad !== nothing
# end

# @testset "oneside optimize mps" for atype in [Array], dtype in [ComplexF64]
#     seed_number = 100
#     β = 0.8
#     D,χ = 2,20
#     mapsteps = 20
#     infolder, outfolder = "./data/", "./data/"

#     model = Ising(β)
#     M = atype{dtype}(model_tensor(model))
#     Random.seed!(seed_number)

#     Au, Ad = optimisemps(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", χ = χ, mapsteps = mapsteps, verbose= true, updown = false)

#     env = obs_env(M,Au,Ad)
#     @test magnetisation(env,model) ≈ magofβ(model) atol=1e-6
#     @test energy(env,model) ≈ eneofβ(model) atol=1e-6
#     # @show energy(env,model)+1.414213779415974 # β = isingβc
# end

# @testset "twoside optimize mps" for atype in [Array], dtype in [ComplexF64]
#     seed_number = 100
#     β = 0.8
#     D,χ = 2,20
#     mapsteps = 20
#     infolder, outfolder = "./data/", "./data/"

#     model = Ising(β)
#     M = atype{dtype}(model_tensor(model))
#     Random.seed!(seed_number)

#     Au, Ad = optimisemps(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", χ = χ, mapsteps = mapsteps, verbose= true, downfromup = true)

#     env = obs_env(M,Au,Ad)
#     @test magnetisation(env,model) ≈ magofβ(model) atol=1e-6
#     @test energy(env,model) ≈ eneofβ(model) atol=1e-6
# end


@testset "Tri-angular AF-Ising model at 0K" for atype in [Array], dtype in [ComplexF64]
    seed_number = 101
    D,χ = 4,20
    mapsteps = 100
    infolder, outfolder = "./data/", "./data/"
    model = "Tri-Ising-Bad-0K"

    # Tri-Ising-bad at 0K
    M = zeros((2,2,2,2))
    M[1,2,1,1]=1.0
    M[2,1,1,1]=1.0
    M[2,2,1,1]=1.0
    M[1,2,2,2]=1.0
    M[2,1,2,2]=1.0
    M[1,1,2,2]=1.0
    # M += rand(2,2,2,2)*0.1

    # Double Layer ?
    # M = reshape(ein"ijkl,abcj->iabkcl"(M,M),4,2,4,2)
    # M = permutedims(M,(2,3,4,1))
    # M = reshape(ein"ijkl,abcj->iabkcl"(M,M),4,4,4,4)

    M = atype{dtype}(M)
    Random.seed!(seed_number)

    Au, Ad = optimisemps(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", χ = χ, mapsteps = mapsteps, verbose= true, downfromup = true)

    env = obs_env(M,Au,Ad)
end

# using HDF5
# @testset "Nv=1 Free Fermion PEPS" for atype in [Array], dtype in [ComplexF64]
#     seed_number = 101
#     D,χ = 4,20
#     mapsteps = 100
#     infolder, outfolder = "./data/", "./data/"
#     model = "Nv=1 Free Fermion PEPS"

#     # Tri-Ising-bad at 0K
#     M = h5read("../ffbulk.h5","bulk_tensor")

#     # Double Layer ?
#     # M = reshape(ein"ijkl,abcj->iabkcl"(M,M),4,2,4,2)
#     # M = permutedims(M,(2,3,4,1))
#     # M = reshape(ein"ijkl,abcj->iabkcl"(M,M),4,4,4,4)

#     M = atype{dtype}(M)
#     Random.seed!(seed_number)

#     Au, Ad = optimisemps(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", χ = χ, mapsteps = mapsteps, verbose= true, downfromup = true)

#     env = obs_env(M,Au,Ad)
# end