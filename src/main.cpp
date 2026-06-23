#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef RD_STAGE_NAME
#define RD_STAGE_NAME "unknown"
#endif

namespace {

#ifdef RD_SMOKE_TEST
constexpr int kWidth = 128;
constexpr int kHeight = 128;
constexpr int kSteps = 80;
#elif defined(RD_PREVIEW_TEST)
constexpr int kWidth = 512;
constexpr int kHeight = 512;
constexpr int kSteps = 5000;
#else
constexpr int kWidth = 2048;
constexpr int kHeight = 2048;
constexpr int kSteps = 5000;
#endif

#ifndef RD_FEED
#define RD_FEED 0.042
#endif
#ifndef RD_KILL
#define RD_KILL 0.059
#endif

constexpr double kDiffusionU = 0.16;
constexpr double kDiffusionV = 0.08;
constexpr double kFeed = RD_FEED;
constexpr double kKill = RD_KILL;
constexpr double kTimeStep = 1.0;
constexpr double kPi = 3.14159265358979323846;
constexpr char kOutputFile[] = "reaction_diffusion.bmp";

struct Cell {
  double u;
  double v;
};

static_assert(std::is_standard_layout_v<Cell>);
static_assert(sizeof(Cell) == 2 * sizeof(double));

struct Rgb {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

int rows_for_rank(int rank, int world_size) {
  return kHeight / world_size + (rank < kHeight % world_size ? 1 : 0);
}

int first_row_for_rank(int rank, int world_size) {
  return rank * (kHeight / world_size) + std::min(rank, kHeight % world_size);
}

std::size_t cell_index(int local_row, int column) {
  return static_cast<std::size_t>(local_row) * kWidth + column;
}

std::uint32_t coordinate_hash(std::uint32_t x, std::uint32_t y) {
  std::uint32_t value = x * 0x8da6b343U ^ y * 0xd8163841U;
  value ^= value >> 13;
  value *= 0x85ebca6bU;
  value ^= value >> 16;
  return value;
}

void initialize_grid(std::vector<Cell> &grid, int first_row, int local_rows) {
  std::fill(grid.begin(), grid.end(), Cell{1.0, 0.0});
  for (int local_row = 1; local_row <= local_rows; ++local_row) {
    const int global_row = first_row + local_row - 1;
    for (int column = 0; column < kWidth; ++column) {
      const std::uint32_t hash = coordinate_hash(column, global_row);
      const double noise_u = static_cast<double>(hash & 0xffffU) / 65535.0;
      const double noise_v =
          static_cast<double>((hash >> 16) & 0xffffU) / 65535.0;
      grid[cell_index(local_row, column)] =
          {0.46 + 0.08 * noise_u, 0.21 + 0.08 * noise_v};
    }
  }
}

void exchange_halos(std::vector<Cell> &grid, int local_rows, int rank,
                    int world_size, MPI_Comm communicator) {
  const int rank_above = (rank - 1 + world_size) % world_size;
  const int rank_below = (rank + 1) % world_size;

  MPI_Sendrecv(grid.data() + cell_index(1, 0), 2 * kWidth, MPI_DOUBLE,
               rank_above, 10,
               grid.data() + cell_index(local_rows + 1, 0), 2 * kWidth,
               MPI_DOUBLE, rank_below, 10, communicator, MPI_STATUS_IGNORE);
  MPI_Sendrecv(grid.data() + cell_index(local_rows, 0), 2 * kWidth, MPI_DOUBLE,
               rank_below, 11, grid.data() + cell_index(0, 0), 2 * kWidth,
               MPI_DOUBLE, rank_above, 11, communicator, MPI_STATUS_IGNORE);
}

void update_grid(const std::vector<Cell> &current, std::vector<Cell> &next,
                 int local_rows) {
  for (int row = 1; row <= local_rows; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      const int left_column = column == 0 ? kWidth - 1 : column - 1;
      const int right_column = column == kWidth - 1 ? 0 : column + 1;
      const Cell &center = current[cell_index(row, column)];
      const Cell &left = current[cell_index(row, left_column)];
      const Cell &right = current[cell_index(row, right_column)];
      const Cell &above = current[cell_index(row - 1, column)];
      const Cell &below = current[cell_index(row + 1, column)];
      const Cell &above_left = current[cell_index(row - 1, left_column)];
      const Cell &above_right = current[cell_index(row - 1, right_column)];
      const Cell &below_left = current[cell_index(row + 1, left_column)];
      const Cell &below_right = current[cell_index(row + 1, right_column)];

      const double laplacian_u =
          -center.u + 0.20 * (left.u + right.u + above.u + below.u) +
          0.05 * (above_left.u + above_right.u + below_left.u + below_right.u);
      const double laplacian_v =
          -center.v + 0.20 * (left.v + right.v + above.v + below.v) +
          0.05 * (above_left.v + above_right.v + below_left.v + below_right.v);
      const double reaction = center.u * center.v * center.v;

      next[cell_index(row, column)].u =
          center.u +
          (kDiffusionU * laplacian_u - reaction + kFeed * (1.0 - center.u)) *
              kTimeStep;
      next[cell_index(row, column)].v =
          center.v +
          (kDiffusionV * laplacian_v + reaction -
           (kFeed + kKill) * center.v) *
              kTimeStep;
    }
  }
}

void simulate(std::vector<Cell> &current, std::vector<Cell> &next,
              int local_rows, int rank, int world_size,
              MPI_Comm communicator) {
  for (int step = 0; step < kSteps; ++step) {
    exchange_halos(current, local_rows, rank, world_size, communicator);
    update_grid(current, next, local_rows);
    current.swap(next);
  }
}

std::vector<Cell> gather_grid(const std::vector<Cell> &local_grid,
                              int local_rows, int rank, int world_size,
                              MPI_Comm communicator) {
  std::vector<int> receive_counts(world_size);
  std::vector<int> receive_offsets(world_size);
  for (int process = 0; process < world_size; ++process) {
    receive_counts[process] = 2 * rows_for_rank(process, world_size) * kWidth;
    receive_offsets[process] =
        2 * first_row_for_rank(process, world_size) * kWidth;
  }

  std::vector<Cell> global_grid;
  if (rank == 0) {
    global_grid.resize(static_cast<std::size_t>(kWidth) * kHeight);
  }
  MPI_Gatherv(local_grid.data() + cell_index(1, 0), 2 * local_rows * kWidth,
              MPI_DOUBLE, rank == 0 ? global_grid.data() : nullptr,
              receive_counts.data(), receive_offsets.data(), MPI_DOUBLE, 0,
              communicator);
  return global_grid;
}

double clamp01(double value) { return std::clamp(value, 0.0, 1.0); }

Rgb color_for(double value) {
  const double intensity = std::sqrt(clamp01(value * 2.8));
  const double phase = 0.72 * intensity;
  const double red =
      intensity * (0.5 + 0.5 * std::cos(2.0 * kPi * (phase + 0.00)));
  const double green =
      intensity * (0.5 + 0.5 * std::cos(2.0 * kPi * (phase + 0.18)));
  const double blue =
      intensity * (0.5 + 0.5 * std::cos(2.0 * kPi * (phase + 0.38)));
  return {
      static_cast<unsigned char>(255.0 * clamp01(red) + 0.5),
      static_cast<unsigned char>(255.0 * clamp01(green) + 0.5),
      static_cast<unsigned char>(255.0 * clamp01(blue) + 0.5),
  };
}

void write_u16(std::ofstream &output, std::uint16_t value) {
  output.put(static_cast<char>(value & 0xff));
  output.put(static_cast<char>((value >> 8) & 0xff));
}

void write_u32(std::ofstream &output, std::uint32_t value) {
  output.put(static_cast<char>(value & 0xff));
  output.put(static_cast<char>((value >> 8) & 0xff));
  output.put(static_cast<char>((value >> 16) & 0xff));
  output.put(static_cast<char>((value >> 24) & 0xff));
}

void write_bmp(const std::vector<Cell> &grid) {
  const std::uint32_t row_bytes =
      static_cast<std::uint32_t>((kWidth * 3 + 3) & ~3);
  const std::uint32_t image_bytes = row_bytes * kHeight;
  std::ofstream output(kOutputFile, std::ios::binary);
  if (!output) {
    throw std::runtime_error(std::string("cannot open ") + kOutputFile);
  }

  output.put('B');
  output.put('M');
  write_u32(output, 54 + image_bytes);
  write_u16(output, 0);
  write_u16(output, 0);
  write_u32(output, 54);
  write_u32(output, 40);
  write_u32(output, kWidth);
  write_u32(output, kHeight);
  write_u16(output, 1);
  write_u16(output, 24);
  write_u32(output, 0);
  write_u32(output, image_bytes);
  write_u32(output, 2835);
  write_u32(output, 2835);
  write_u32(output, 0);
  write_u32(output, 0);

  const unsigned char padding[3] = {0, 0, 0};
  const int padding_size = static_cast<int>(row_bytes) - kWidth * 3;
  for (int row = kHeight - 1; row >= 0; --row) {
    for (int column = 0; column < kWidth; ++column) {
      const Rgb color = grid[static_cast<std::size_t>(row) * kWidth + column].v >
                                0.0
                            ? color_for(grid[static_cast<std::size_t>(row) *
                                                 kWidth + column]
                                            .v)
                            : Rgb{0, 0, 0};
      output.put(static_cast<char>(color.blue));
      output.put(static_cast<char>(color.green));
      output.put(static_cast<char>(color.red));
    }
    output.write(reinterpret_cast<const char *>(padding), padding_size);
  }
  if (!output) {
    throw std::runtime_error(std::string("failed while writing ") +
                             kOutputFile);
  }
}

std::uint64_t checksum(const std::vector<Cell> &grid) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const Cell &cell : grid) {
    const Rgb color = color_for(cell.v);
    hash ^= color.red;
    hash *= 1099511628211ULL;
    hash ^= color.green;
    hash *= 1099511628211ULL;
    hash ^= color.blue;
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (argc != 1 || world_size > kHeight) {
    if (rank == 0) {
      std::cerr << "error: use only mpirun parameters; MPI ranks must not exceed "
                << kHeight << '\n';
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  const int local_rows = rows_for_rank(rank, world_size);
  const int first_row = first_row_for_rank(rank, world_size);
  std::vector<Cell> current(
      static_cast<std::size_t>(local_rows + 2) * kWidth);
  std::vector<Cell> next(current.size());
  initialize_grid(current, first_row, local_rows);

  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  simulate(current, next, local_rows, rank, world_size, MPI_COMM_WORLD);
  const double local_seconds = MPI_Wtime() - start;
  double seconds = 0.0;
  MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  std::vector<Cell> global_grid =
      gather_grid(current, local_rows, rank, world_size, MPI_COMM_WORLD);
  int exit_code = EXIT_SUCCESS;
  if (rank == 0) {
    try {
      write_bmp(global_grid);
      std::cout << "stage=" << RD_STAGE_NAME << " ranks=" << world_size
                << " threads_per_rank=1 resolution=" << kWidth << 'x'
                << kHeight << " steps=" << kSteps << " seconds=" << std::fixed
                << std::setprecision(3) << seconds << " output=" << kOutputFile
                << " checksum=0x" << std::hex << checksum(global_grid)
                << std::dec << '\n';
    } catch (const std::exception &error) {
      std::cerr << "error: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
    }
  }

  MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return exit_code;
}
