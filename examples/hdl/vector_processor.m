function [mag_sq, dot_prod] = vector_processor(vec_a, vec_b)
    %#codegen
    
    % vec_a, vec_b: Vetores de 3 elementos (16 bits, 8 frac)
    % mag_sq: Magnitude ao quadrado de vec_a
    % dot_prod: Produto escalar entre a e b
    % Garante que as entradas sejam tratadas como fi 16,8
    vec_a = fi(vec_a, 1, 16, 8);
    vec_b = fi(vec_b, 1, 16, 8);
    
    % Multiplicações (16x16 bits = 32 bits internos)
    p1 = vec_a(1) * vec_b(1);
    p2 = vec_a(2) * vec_b(2);
    p3 = vec_a(3) * vec_b(3);
    
    % Produto Escalar (Soma de produtos)
    dot_prod = p1 + p2 + p3;
    
    % Magnitude ao quadrado (A(1)^2 + A(2)^2 + A(3)^2)
    a1_sq = vec_a(1) * vec_a(1);
    a2_sq = vec_a(2) * vec_a(2);
    a3_sq = vec_a(3) * vec_a(3);
    
    mag_sq = a1_sq + a2_sq + a3_sq;
end

