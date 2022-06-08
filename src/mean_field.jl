function hamiltonian!(ds, s, J::AbstractMatrix, h::AbstractVector, anisotropy, t)
    for i in 1:size(s, 1) # iterate over all spins i
        js_x, js_y, js_z = @uviews J s begin
            js_x = 0.5 * anisotropy[1] * get_mean_field(J, s, i, 1) + h[1]
            js_y = 0.5 * anisotropy[2] * get_mean_field(J, s, i, 2) + h[2]
            js_z = 0.5 * anisotropy[3] * get_mean_field(J, s, i, 3) + h[3]
            js_x, js_y, js_z
        end

        ds[i, 1] = s[i, 2] * js_z - s[i, 3] * js_y
        ds[i, 2] = s[i, 3] * js_x - s[i, 1] * js_z
        ds[i, 3] = s[i, 1] * js_y - s[i, 2] * js_x
    end
    nothing
end

function hamiltonian!(ds, s, J::AbstractMatrix, h::AbstractMatrix, anisotropy, t)
    for i in 1:size(s, 1) # iterate over all spins i
        js_x, js_y, js_z = @uviews J s begin
            js_x = 0.5 * anisotropy[1] * get_mean_field(J, s, i, 1) + h[i, 1]
            js_y = 0.5 * anisotropy[2] * get_mean_field(J, s, i, 2) + h[i, 2]
            js_z = 0.5 * anisotropy[3] * get_mean_field(J, s, i, 3) + h[i, 3]
            js_x, js_y, js_z
        end

        ds[i, 1] = s[i, 2] * js_z - s[i, 3] * js_y
        ds[i, 2] = s[i, 3] * js_x - s[i, 1] * js_z
        ds[i, 3] = s[i, 1] * js_y - s[i, 2] * js_x
    end
    nothing
end

get_mean_field(J_i::AbstractVector{T}, s::AbstractVector{T}) where T = J_i ⋅ s

function get_mean_field(J::AbstractMatrix{T}, s::AbstractMatrix{T}, i, α) where T
    get_mean_field(view(J, :, i), view(s, :, α))
end

function get_mean_field_unsafe(J::AbstractMatrix{T}, s::AbstractMatrix{T}, i, α) where T
    s_α = uview(s, :, α)
    J_i = uview(J, :, i)
    get_mean_field(J_i, s_α)
end
