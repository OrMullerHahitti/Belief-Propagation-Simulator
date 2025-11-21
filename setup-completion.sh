#!/bin/bash
# Setup script for Makefile completion

# Function to detect the current shell
detect_shell() {
    if [[ -n "$ZSH_VERSION" ]]; then
        echo "zsh"
    elif [[ -n "$BASH_VERSION" ]]; then
        echo "bash"
    else
        echo "unknown"
    fi
}

# Function to setup completion for current session
setup_completion() {
    local shell_type=$(detect_shell)
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    case "$shell_type" in
        "zsh")
            echo "Setting up zsh completion..."
            source "$script_dir/.make-completion.zsh"
            echo "✓ Zsh completion loaded for current session"
            ;;
        "bash")
            echo "Setting up bash completion..."
            source "$script_dir/.make-completion.bash"
            echo "✓ Bash completion loaded for current session"
            ;;
        *)
            echo "❌ Unsupported shell: $shell_type"
            return 1
            ;;
    esac
}

# Function to show permanent setup instructions
show_permanent_setup() {
    local shell_type=$(detect_shell)
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    echo ""
    echo "To make completion permanent, add the following to your shell config:"
    echo ""
    
    case "$shell_type" in
        "zsh")
            echo "# Add to ~/.zshrc:"
            echo "source \"$script_dir/.make-completion.zsh\""
            ;;
        "bash")
            echo "# Add to ~/.bashrc or ~/.bash_profile:"
            echo "source \"$script_dir/.make-completion.bash\""
            ;;
        *)
            echo "Shell not detected. Please manually source the appropriate completion file:"
            echo "  - For bash: source \"$script_dir/.make-completion.bash\""
            echo "  - For zsh: source \"$script_dir/.make-completion.zsh\""
            ;;
    esac
    
    echo ""
    echo "Available make targets:"
    if [[ -f Makefile ]]; then
        grep -E '^[a-zA-Z0-9_-]+:.*##' Makefile | sort | awk 'BEGIN {FS=":.*## "} {printf "  %-18s %s\n", $1, $2}'
    else
        echo "  No Makefile found in current directory"
    fi
}

# Main execution
main() {
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [--permanent]"
        echo ""
        echo "Options:"
        echo "  --permanent    Show instructions for permanent setup"
        echo "  --help, -h     Show this help message"
        echo ""
        echo "Without options, sets up completion for current session only."
        return 0
    fi
    
    setup_completion
    
    if [[ "$1" == "--permanent" ]]; then
        show_permanent_setup
    else
        echo ""
        echo "💡 Run '$0 --permanent' to see instructions for permanent setup"
        echo ""
        echo "Try: make <TAB><TAB> to see available targets"
    fi
}

# Run main function with all arguments
main "$@"
