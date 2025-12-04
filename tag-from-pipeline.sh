#!/bin/bash

function version_gt() {
    test "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1";
}

function get_version_in_changelog() {
    # Look for version pattern like ## [0.0.1] - 30.04.2025
    version_line=$(grep -m 1 "## \[.*\]" CHANGELOG.md)

    if [[ -z "$version_line" ]]; then
        printf "\033[0;31mChangelog file is incorrect, could not find a line matching '## [x.x.x]' format.\033[0m"
        return 171
    fi

    # Extract version number from between brackets
    if [[ $version_line =~ \[([0-9]+\.[0-9]+\.[0-9]+)\] ]]; then
        version="${BASH_REMATCH[1]}"
        # Extract date if exists
        if [[ $version_line =~ \[([0-9]+\.[0-9]+\.[0-9]+)\]\ -\ ([0-9.]+) ]]; then
            date="${BASH_REMATCH[2]}"
            echo "$version $date"
        else
            echo "$version"
        fi
        return
    else
        printf "\033[0;31mChangelog file is incorrect, version should match '[x.x.x]' format.\033[0m"
        return 171
    fi
}

function verify_changelog_version() {
    read -r version date < <(get_version_in_changelog)

    if [ $? -ne 0 ]; then
        return 171
    fi

    current_version=$(git tag -l --sort=-version:refname | grep -E "^[0-9]+(\.[0-9]+){1,2}$" | head -n 1)

    if [ -z "$current_version" ]; then
        printf "No existing version tags found. New version is %s.\n" "$version"
        return 0
    fi

    if ! version_gt "$version" "$current_version"; then
        printf "\033[0;31mNew version in the changelog (%s) should be greater than the current version (%s).\n\033[0m" "$version" "$current_version"
        return 172
    fi

    if [ $(git tag -l "$version") ]; then
        printf "Version %s already exists.\n" "$version"
        return 172
    fi

    printf "Current version is %s, new version is %s.\n" "$current_version" "$version"
}

function create_new_tag() {
    read -r version date < <(get_version_in_changelog)

    if [ $? -ne 0 ]; then
        return 171
    fi

    # Only tag if version doesn't exist yet
    if ! [ $(git tag -l "$version") ]; then
        printf "Releasing version %s.\n" "$version"
        curl -X POST -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/$GITHUB_REPOSITORY/releases" \
        -d "{\"tag_name\": \"$version\", \"name\": \"$version\", \"body\": \"Changelog: https://github.com/$GITHUB_REPOSITORY/blob/main/CHANGELOG.md\"}"
    else
        printf "Version %s already exists, not creating a new tag.\n" "$version"
    fi
}

"$@"
