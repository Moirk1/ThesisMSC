import struct
import csv

def parse_header(file_path):
    """
    Reads the text header of the RTD file, extracts HeaderSize, Gain, and returns byte offset.
    """
    header_lines = 0
    gain = 1.0
    altitudes = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        byte_offset = 0
        first_line = f.readline()
        byte_offset += len(first_line.encode('utf-8'))
        print(f"[DEBUG] First line: {first_line.strip()}, byte offset: {byte_offset}")

        if first_line.startswith("HeaderSize="):
            header_lines = int(first_line.strip().split('=')[1])
        else:
            raise ValueError("File does not start with 'HeaderSize='")

        for i in range(header_lines):
            line = f.readline()
            byte_offset += len(line.encode('utf-8'))
            print(f"[DEBUG] Line {i+1}: {line.strip()}, byte offset: {byte_offset}")

            if "Gain=" in line:
                gain = float(line.strip().split('=')[1])

            if "Altitudes(m)=" in line:
                altitudes = [int(x) for x in line.strip().split('=')[1].split()]

    # We return the byte offset as the position right after the header
    return byte_offset, gain, altitudes


def unpack_time_offset(binary_chunk):
    """
    Unpacks the time offset from the binary data as a signed 32-bit integer (big-endian).
    """
    time_offset = struct.unpack(">i", binary_chunk[:4])[0]
    return time_offset  # Already in 1/10 sec


def read_and_unpack_time_offsets(filepath):
    """
    Reads time offsets from the RTD file.
    """
    time_offsets = []

    # Step 1: Parse header to get byte offset + gain
    byte_offset, gain, _ = parse_header(filepath)

    with open(filepath, 'rb') as f:
        print(f"[DEBUG] Seeking to byte offset: {byte_offset}")
        f.seek(byte_offset)

        record_size = 15 +  (20*12) # Adjust based on HeightCount

        record_index = 0
        while True:
            binary_chunk = f.read(record_size)
            if len(binary_chunk) < record_size:
                break

            # Print the first 4 bytes of the current block
            #print(f"[DEBUG] First 4 bytes of record {record_index}: {binary_chunk[:4].hex()}")
            #print(f"[DEBUG] Full binary chunk (record {record_index}): {binary_chunk.hex()}")
            # Unpack the time offset for the current record
            time_offset = unpack_time_offset(binary_chunk)
            time_offsets.append(time_offset)

             # Print the full binary chunk for the first three records
            if record_index < 3:
                print(f"[DEBUG] Full binary chunk (record {record_index}): {binary_chunk.hex()}")
                print(f"[DEBUG] First 4 bytes of record {record_index}: {binary_chunk[:4].hex()}")
                


            record_index += 1

    print(f"[INFO] Total records read: {len(time_offsets)}")
    return time_offsets


def save_time_offsets_to_csv(time_offsets, output_filename="time_offsets.csv"):
    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["TimeOffset (1/10 sec)"])
        for t in time_offsets:
            writer.writerow([t])
    print(f"[INFO] Saved to {output_filename}")


# === MAIN ===
file_path = "WLS70-001_2022_08_05__00_00_00 (1).rtd"

time_offsets = read_and_unpack_time_offsets(file_path)
save_time_offsets_to_csv(time_offsets)
