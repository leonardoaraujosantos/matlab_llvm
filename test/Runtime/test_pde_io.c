/* test_pde_io.c — round-trip + format tests for STL and GLB importers.
 *
 * Generates small synthetic STL and GLB files programmatically and
 * verifies that matlab_pde_load_stl / matlab_pde_load_glb recover the
 * expected vertex / face counts and coordinates after vertex welding.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct matlab_mat {
    double *data;
    int64_t rows;
    int64_t cols;
} matlab_mat;
typedef struct matlab_struct_s matlab_struct;

extern matlab_struct *matlab_pde_load_stl_path(const char *path, int64_t plen);
#define matlab_pde_load_stl matlab_pde_load_stl_path
extern double matlab_pde_save_stl_binary(matlab_struct *m,
                                          const char *path, int64_t plen);
extern matlab_struct *matlab_pde_load_glb_path(const char *path, int64_t plen);
#define matlab_pde_load_glb matlab_pde_load_glb_path
extern matlab_struct *matlab_pde_mesh_cuboid_tet(double W, double D, double H,
                                                  double Nx, double Ny, double Nz);
extern double matlab_pde_num_nodes(matlab_struct *mesh);
extern double matlab_pde_num_faces(matlab_struct *mesh);
extern matlab_mat *matlab_pde_mesh_nodes(matlab_struct *mesh);
extern matlab_mat *matlab_pde_mesh_faces(matlab_struct *mesh);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); } \
} while (0)

/* Write a tiny ASCII STL describing a single triangle. */
static void write_ascii_stl_tri(const char *path) {
    FILE *f = fopen(path, "w");
    fprintf(f,
        "solid tri\n"
        " facet normal 0 0 1\n"
        "  outer loop\n"
        "   vertex 0 0 0\n"
        "   vertex 1 0 0\n"
        "   vertex 0 1 0\n"
        "  endloop\n"
        " endfacet\n"
        "endsolid tri\n");
    fclose(f);
}

/* Write a binary STL describing a regular tetrahedron — 4 triangles
 * with 4 unique vertices, vertex welding should collapse the 12
 * per-triangle vertex records into 4 distinct nodes. */
static void write_binary_stl_tet(const char *path) {
    FILE *f = fopen(path, "wb");
    char hdr[80] = {0};
    memcpy(hdr, "tet binary", 10);
    fwrite(hdr, 1, 80, f);
    uint32_t ntri = 4;
    fwrite(&ntri, 4, 1, f);
    /* 4 vertices of a regular tetrahedron */
    float V[4][3] = {
        { 0.f,    0.f,    0.f },
        { 1.f,    0.f,    0.f },
        { 0.5f,   0.866f, 0.f },
        { 0.5f,   0.289f, 0.816f },
    };
    int faces[4][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
    };
    for (int i = 0; i < 4; ++i) {
        float nrm[3] = {0, 0, 0};
        fwrite(nrm, 4, 3, f);
        for (int k = 0; k < 3; ++k) fwrite(V[faces[i][k]], 4, 3, f);
        uint16_t attr = 0;
        fwrite(&attr, 2, 1, f);
    }
    fclose(f);
}

/* Write a minimal GLB with a single primitive: a unit-quad split into
 * 2 triangles. 4 vertices, 6 indices (uint16). */
static void write_glb_quad(const char *path) {
    /* JSON describing the asset. */
    const char *json =
        "{"
        "\"asset\":{\"version\":\"2.0\"},"
        "\"meshes\":[{\"primitives\":[{"
            "\"attributes\":{\"POSITION\":0},"
            "\"indices\":1,"
            "\"mode\":4"
        "}]}],"
        "\"accessors\":["
            "{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"},"
            "{\"bufferView\":1,\"componentType\":5123,\"count\":6,\"type\":\"SCALAR\"}"
        "],"
        "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48},"
            "{\"buffer\":0,\"byteOffset\":48,\"byteLength\":12}"
        "],"
        "\"buffers\":[{\"byteLength\":60}]"
        "}";
    /* Pad JSON to a multiple of 4 with spaces. */
    size_t jlen = strlen(json);
    while (jlen % 4 != 0) jlen++;
    char *jbuf = (char *)malloc(jlen);
    memcpy(jbuf, json, strlen(json));
    for (size_t i = strlen(json); i < jlen; ++i) jbuf[i] = ' ';

    /* BIN: 4 vec3 floats + 6 uint16 indices. */
    uint8_t bin[60];
    float pos[12] = {
        0.f, 0.f, 0.f,
        1.f, 0.f, 0.f,
        1.f, 1.f, 0.f,
        0.f, 1.f, 0.f,
    };
    memcpy(bin, pos, 48);
    uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    memcpy(bin + 48, idx, 12);

    FILE *f = fopen(path, "wb");
    /* Header. */
    uint8_t hdr[12];
    memcpy(hdr, "glTF", 4);
    uint32_t version = 2;
    memcpy(hdr + 4, &version, 4);
    uint32_t total = 12 + 8 + (uint32_t)jlen + 8 + 60;
    memcpy(hdr + 8, &total, 4);
    fwrite(hdr, 1, 12, f);
    /* JSON chunk. */
    uint32_t cl = (uint32_t)jlen;
    uint32_t ct = 0x4E4F534A;  /* "JSON" */
    fwrite(&cl, 4, 1, f);
    fwrite(&ct, 4, 1, f);
    fwrite(jbuf, 1, jlen, f);
    /* BIN chunk. */
    cl = 60;
    ct = 0x004E4942;  /* "BIN\0" */
    fwrite(&cl, 4, 1, f);
    fwrite(&ct, 4, 1, f);
    fwrite(bin, 1, 60, f);
    fclose(f);
    free(jbuf);
}

static void test_stl_ascii(void) {
    printf("  test_stl_ascii:\n");
    const char *path = "/tmp/test_pde_tri.stl";
    write_ascii_stl_tri(path);
    matlab_struct *m = matlab_pde_load_stl(path, (int64_t)strlen(path));
    CHECK(m, "load_stl returned NULL");
    int Nn = (int)matlab_pde_num_nodes(m);
    int Nf = (int)matlab_pde_num_faces(m);
    printf("    triangle: %d nodes, %d faces\n", Nn, Nf);
    CHECK(Nn == 3, "ascii triangle should have 3 unique nodes");
    CHECK(Nf == 1, "ascii triangle should have 1 face");
}

static void test_stl_binary(void) {
    printf("  test_stl_binary:\n");
    const char *path = "/tmp/test_pde_tet.stl";
    write_binary_stl_tet(path);
    matlab_struct *m = matlab_pde_load_stl(path, (int64_t)strlen(path));
    CHECK(m, "load_stl returned NULL");
    int Nn = (int)matlab_pde_num_nodes(m);
    int Nf = (int)matlab_pde_num_faces(m);
    printf("    tetrahedron: %d nodes (welded from 12), %d faces\n", Nn, Nf);
    CHECK(Nn == 4, "binary tetrahedron should weld to 4 nodes");
    CHECK(Nf == 4, "binary tetrahedron should have 4 faces");
}

static void test_stl_roundtrip(void) {
    printf("  test_stl_roundtrip:\n");
    /* Build a cuboid mesh's surface, write it as STL, then load back. */
    matlab_struct *src = matlab_pde_mesh_cuboid_tet(1, 1, 1, 2, 2, 2);
    int Nn_src = (int)matlab_pde_num_nodes(src);
    matlab_mat *faces = matlab_pde_mesh_faces(src);
    int Nf_total = (int)faces->rows;
    const char *path = "/tmp/test_pde_cuboid.stl";
    CHECK(matlab_pde_save_stl_binary(src, path, (int64_t)strlen(path)) != 0.0,
          "save_stl_binary failed");
    matlab_struct *dst = matlab_pde_load_stl(path, (int64_t)strlen(path));
    CHECK(dst, "load_stl on round-tripped file returned NULL");
    int Nn_dst = (int)matlab_pde_num_nodes(dst);
    int Nf_dst = (int)matlab_pde_num_faces(dst);
    printf("    saved+loaded cuboid: %d/%d nodes, %d/%d faces\n",
           Nn_dst, Nn_src, Nf_dst, Nf_total);
    /* STL stores only surface triangles — count must match the source
     * surface face count, and the unique vertices on the surface
     * (corner + face + edge nodes; for a 2x2x2 cuboid that's 26). */
    CHECK(Nf_dst == Nf_total, "round-tripped STL face count mismatch");
    CHECK(Nn_dst == 26, "round-tripped STL should have 26 surface vertices");
}

static void test_glb(void) {
    printf("  test_glb:\n");
    const char *path = "/tmp/test_pde_quad.glb";
    write_glb_quad(path);
    matlab_struct *m = matlab_pde_load_glb(path, (int64_t)strlen(path));
    CHECK(m, "load_glb returned NULL");
    int Nn = (int)matlab_pde_num_nodes(m);
    int Nf = (int)matlab_pde_num_faces(m);
    printf("    quad: %d nodes, %d faces\n", Nn, Nf);
    CHECK(Nn == 4, "glb quad should have 4 nodes");
    CHECK(Nf == 2, "glb quad should have 2 triangles");
}

int main(void) {
    printf("test_pde_io:\n");
    test_stl_ascii();
    test_stl_binary();
    test_stl_roundtrip();
    test_glb();
    printf("all tests passed.\n");
    return 0;
}
