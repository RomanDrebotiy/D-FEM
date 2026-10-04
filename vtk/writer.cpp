#include "writer.h"
#include "../mesh/mpi_build_mesh.h"
#include "../fem/per_block_global.h"
#include "../mesh/geometry.h"
#include <fstream>

int write_vtk(BlockMesh& bm, GlobalVector& v, int rank, int size) {
    std::string filename = "solution_vtk/solution_" + std::to_string(rank) + ".vtu";
    std::ofstream file(filename);
    int num_cells = bm.triangles.size();
    int num_nodes = v.get_size();

    file << R"(<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
)" << "    <Piece NumberOfPoints=\"" << num_nodes << "\" NumberOfCells=\"" << num_cells << "\">\n"
<< R"(      <PointData Scalars="u">
        <DataArray type="Float64" Name="u" NumberOfComponents="1" format="ascii">
)" << "          ";

    for (double x : v.vals) {
        file << x << " ";
    }

    file << R"(
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
)";

    for (int i = 0; i < 4; i++) {
        file << "          " << bm.corners[i].x << " " << bm.corners[i].y << " 0.0\n";
    }

    for (int i = 0; i < 4; i++) {
        for (size_t k = 1; k < bm.ifaces[i]->nodes.size() - 1; k++) {
            file << "          " << bm.ifaces[i]->nodes[k].x << " " << bm.ifaces[i]->nodes[k].y << " 0.0\n";
        }
    }

    for (size_t i = 0; i < bm.inner_nodes.size(); i++) {
        file << "          " << bm.inner_nodes[i].x << " " << bm.inner_nodes[i].y << " 0.0\n";
    }

    file << R"(        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
)";

    for (Triangle t : bm.triangles) {
        file << "          " 
             << t.verts[0]->block_local_idx << " "
             << t.verts[1]->block_local_idx << " "
             << t.verts[2]->block_local_idx << "\n";
    }

    file << R"(        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
)" << "          ";

    for (int i = 1; i <= num_cells; i++) {
        file << 3 * i << " ";
    }

    file << R"(
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
)" << "          ";

    for (int i = 1; i <= num_cells; i++) {
        file << "5 ";
    }

    file << R"(
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
)";

    file.close();

    if (rank == 0) {
        std::ofstream file("solution_vtk/solution.pvtu");
            file << R"(<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <PUnstructuredGrid GhostLevel="0">
    <PPointData Scalars="u">
      <PDataArray
          type="Float64"
          Name="u"
          NumberOfComponents="1"/>
    </PPointData>
    <PCellData>
    </PCellData>
    <PPoints>
      <PDataArray
          type="Float64"
          NumberOfComponents="3"/>
    </PPoints>
)";

        for (int i = 0; i < size; i++) {
            file << "    <Piece Source=\"solution_" << i << ".vtu\"/>\n";
        }

        file << R"(  </PUnstructuredGrid>
</VTKFile>
)";

        file.close();
    }

    return 0;
}
