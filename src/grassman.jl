using Manifolds,Optim

import Optim.ldiv!, Optim.dot, Optim.project_tangent!, Optim.retract!,Optim.Manifold
export MPSPreconditioner, ldiv!, dot
struct MPSPreconditioner
    matrix::AbstractArray
end
ldiv!(pgr::Matrix, p::MPSPreconditioner, gr::Matrix) = copyto!(pgr, gr / p.matrix)
dot(x::Matrix, p::MPSPreconditioner, y::Matrix) = dot(x, y * p.matrix)

struct Grassmann <: Optim.Manifold
end

function project_tangent!(M::Grassmann,g,x)
    g .= project(Manifolds.Grassmann(size(x)...),x,g)
end

function retract!(M::Grassmann,x) 
    x .= project(Manifolds.Grassmann(size(x)...),x)
end