# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License header checker."""

# /// script
# dependencies = ["fuzzywuzzy"]
# ///

import argparse
import collections
import logging
import sys
from pathlib import Path

from fuzzywuzzy import fuzz

LOGGER = logging.getLogger(__name__)

FUZZY_MATCH_EXTRA_LINES_TO_CHECK = 3

SKIP_LICENSE_INSERTION_COMMENT = "SKIP LICENSE INSERTION"

DEBUG_LEVENSHTEIN_DISTANCE_CALCULATION = False

LicenseInfo = collections.namedtuple(
    "LicenseInfo",
    [
        "prefixed_license",
        "plain_license",
        "eol",
        "comment_start",
        "comment_prefix",
        "comment_end",
        "num_extra_lines",
    ],
)

EXCLUDE_DIRECTORIES = ["tools", ".pytype", ".cache"]
DEFAULT_LICENSE_TEMPLATE = str(Path(__file__).parent / "license_header.template")


def main(argv=None):
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path from which files should be checked")
    parser.add_argument("--files", default="*/**", help="Filenames or pattern that should be checked")
    parser.add_argument("--license-template", default=DEFAULT_LICENSE_TEMPLATE, help="License file template")
    parser.add_argument("--top-lines-count", type=int, default=5, help="Number of top lines to check")
    parser.add_argument("--fuzzy-ratio-cut-off", type=int, default=85, help="Fuzzy ratio cutoff")
    parser.add_argument(
        "--comment-style",
        default="#",
        help="Can be a single prefix or a triplet: <comment-start>|<comment-prefix>|<comment-end>E.g.: /*| *| */",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose logs",
        action="store_true",
        default=False,
    )
    args = parser.parse_args(argv)

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    license_info = _get_license_info(args)

    failed_files = []

    check_failed = _process_files(args, failed_files, license_info)

    if check_failed:
        LOGGER.error("Some sources contain inconsistent licenses:")
        for file in failed_files:
            LOGGER.error(" %s", file)
        return 1
    return 0


def _get_license_info(args):
    """Get license info from the comment style.

    Args:
        args: arguments of the hook

    Returns:
        LicenseInfo named tuple containing information about the license
    """
    if "|" in args.comment_style:
        comment_start, comment_prefix, comment_end = args.comment_style.split("|")
    else:
        comment_start, comment_prefix, comment_end = None, args.comment_style, None
    with open(args.license_template, encoding="utf8") as license_file:
        plain_license = license_file.readlines()
    prefixed_license = ["{}{}{}".format(comment_prefix, " " if line.strip() else "", line) for line in plain_license]
    eol = "\r\n" if prefixed_license[0][-2:] == "\r\n" else "\n"

    num_extra_lines = 0

    if not prefixed_license[-1].endswith(eol):
        prefixed_license[-1] += eol
        num_extra_lines += 1
    if comment_start:
        prefixed_license = [comment_start + eol] + prefixed_license
        num_extra_lines += 1
    if comment_end:
        prefixed_license = prefixed_license + [comment_end + eol]
        num_extra_lines += 1

    license_info = LicenseInfo(
        prefixed_license=prefixed_license,
        plain_license=plain_license,
        eol=eol,
        comment_start=comment_start,
        comment_prefix=comment_prefix,
        comment_end=comment_end,
        num_extra_lines=num_extra_lines,
    )
    return license_info


def _get_front_matter_end(src_file_content):
    """Return the line index after YAML front matter, or 0 if none present."""
    if not src_file_content or src_file_content[0].strip() != "---":
        return 0
    for i in range(1, len(src_file_content)):
        if src_file_content[i].strip() == "---":
            return i + 1
    return 0


def _process_files(args, failed_files, license_info):
    """Processes all license files.

    Args:
        args: arguments of the hook
        failed_files: list of files that license check failed
        license_info: license info named tuple

    Returns:
        True if some files were changed or t.o.d.o is detected
    """
    workdir = Path(args.path)
    files = workdir.rglob(args.files)
    for src_filepath in files:
        exclude = False
        for excluded in EXCLUDE_DIRECTORIES:
            try:
                src_filepath.relative_to(workdir / excluded)
                LOGGER.debug("The %s is part of excluded %s directory. Skipping.", src_filepath, excluded)
                exclude = True
                break
            except ValueError:
                pass

        if exclude:
            continue

        if Path(src_filepath).is_symlink() or not src_filepath.is_file():
            continue

        LOGGER.info("Processing file: %s", src_filepath)
        with src_filepath.open("r") as src_file:
            src_file_content = src_file.readlines()

        if not _contains_license_header(src_file_content, src_filepath, license_info, args):
            failed_files.append(src_filepath)

    return failed_files


def _contains_license_header(src_file_content, src_filepath, license_info, args):
    """Check if the file contains a license header.

    Args:
        src_file_content: the content of the file
        src_filepath: the path to the file
        license_info: license info named tuple
        args: arguments of the hook
    """
    candidate_contents = [src_file_content]
    if src_filepath.suffix == ".md":
        front_matter_end = _get_front_matter_end(src_file_content)
        if front_matter_end:
            candidate_contents.append(src_file_content[front_matter_end:])

    for content_to_check in candidate_contents:
        license_header_index = _find_license_header_index(
            src_file_content=content_to_check,
            license_info=license_info,
            top_lines_count=args.top_lines_count,
        )

        fuzzy_license_header_index = _fuzzy_find_license_header_index(
            src_file_content=content_to_check,
            license_info=license_info,
            top_lines_count=args.top_lines_count,
            fuzzy_match_extra_lines_to_check=FUZZY_MATCH_EXTRA_LINES_TO_CHECK,
            fuzzy_ratio_cut_off=args.fuzzy_ratio_cut_off,
        )

        if license_header_index is not None or fuzzy_license_header_index is not None:
            return True

    return False


def _find_license_header_index(src_file_content, license_info, top_lines_count):
    """Find the line number where the license header comment starts in the file.

    Args:
        src_file_content: the content of the file
        license_info: license info named tuple
        top_lines_count: the number of top lines to check

    Returns:
        The line number where the license header comment starts in the file, or None if not found.
    """
    for i in range(top_lines_count):
        license_match = True
        for j, license_line in enumerate(license_info.prefixed_license):
            if i + j >= len(src_file_content) or license_line.strip() != src_file_content[i + j].strip():
                license_match = False
                break
        if license_match:
            return i
    return None


def _fuzzy_find_license_header_index(
    src_file_content,  # pylint: disable=too-many-locals
    license_info,
    top_lines_count,
    fuzzy_match_extra_lines_to_check,
    fuzzy_ratio_cut_off,
):
    """Find the line number where the fuzzy matching found best match with ratio higher than the cutoff ratio.

    Args:
        src_file_content: the content of the file
        license_info: license info named tuple
        top_lines_count: the number of top lines to check
        fuzzy_match_extra_lines_to_check: the number of extra lines to check
        fuzzy_ratio_cut_off: the cutoff ratio
    """
    best_line_number_match = None
    best_ratio = 0
    best_num_token_diff = 0
    license_string = " ".join(license_info.plain_license).replace("\n", "").replace("\r", "").strip()
    expected_num_tokens = len(license_string.split(" "))
    for i in range(top_lines_count):
        candidate_array = src_file_content[
            i : i + len(license_info.plain_license) + license_info.num_extra_lines + fuzzy_match_extra_lines_to_check
        ]
        license_string_candidate, candidate_offset = _get_license_candidate_string(candidate_array, license_info)
        ratio = fuzz.token_set_ratio(license_string, license_string_candidate)
        num_tokens = len(license_string_candidate.split(" "))
        num_tokens_diff = abs(num_tokens - expected_num_tokens)
        if DEBUG_LEVENSHTEIN_DISTANCE_CALCULATION:
            LOGGER.info("License_string:%s", license_string)
            LOGGER.info("License_string_candidate:%s", license_string_candidate)
            LOGGER.info("Candidate offset:%s", candidate_offset)
            LOGGER.info("Ratio:%s", ratio)
            LOGGER.info("Number of tokens:%s", num_tokens)
            LOGGER.info("Expected number of tokens:%s", expected_num_tokens)
            LOGGER.info("Num tokens diff:%s", num_tokens_diff)
        if ratio >= fuzzy_ratio_cut_off:
            if ratio > best_ratio or (ratio == best_ratio and num_tokens_diff < best_num_token_diff):
                best_ratio = ratio
                best_line_number_match = i + candidate_offset
                best_num_token_diff = num_tokens_diff
                if DEBUG_LEVENSHTEIN_DISTANCE_CALCULATION:
                    LOGGER.info(
                        "Setting best line number match: %s, ratio %s, num tokens diff %s",
                        best_line_number_match,
                        best_ratio,
                        best_num_token_diff,
                    )
        if DEBUG_LEVENSHTEIN_DISTANCE_CALCULATION:
            LOGGER.info("Best offset match %s", best_line_number_match)
    return best_line_number_match


def _get_license_candidate_string(candidate_array, license_info):
    """Get license candidate string from the array of strings retrieved.

    Args:
        candidate_array: array of lines of the candidate strings
        license_info: LicenseInfo named tuple containing information about the license

    Returns:
        Tuple of string version of the license candidate and offset in lines where it starts.
    """
    license_string_candidate = ""
    stripped_comment_start = license_info.comment_start.strip() if license_info.comment_start else ""
    stripped_comment_prefix = license_info.comment_prefix.strip() if license_info.comment_prefix else ""
    stripped_comment_end = license_info.comment_end.strip() if license_info.comment_end else ""
    in_license = False
    current_offset = 0
    found_license_offset = 0
    for license_line in candidate_array:
        stripped_line = license_line.strip()
        if not in_license:
            if stripped_comment_start:
                if stripped_line.startswith(stripped_comment_start):
                    in_license = True
                    found_license_offset = current_offset + 1  # License starts in the next line
                    continue
            else:
                if stripped_comment_prefix:
                    if stripped_line.startswith(stripped_comment_prefix):
                        in_license = True
                        found_license_offset = current_offset  # License starts in this line
                else:
                    in_license = True
                    found_license_offset = current_offset  # We have no data :(. We start license immediately
        else:
            if stripped_comment_end and stripped_line.startswith(stripped_comment_end):
                break
        if in_license and (not stripped_comment_prefix or stripped_line.startswith(stripped_comment_prefix)):
            license_string_candidate += stripped_line[len(stripped_comment_prefix) :] + " "
        current_offset += 1
    return license_string_candidate.strip(), found_license_offset


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
