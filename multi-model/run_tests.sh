#!/bin/bash
# Quick test runner script for the comprehensive test suite

set -e

echo "=================================================="
echo "Multi-Model Comprehensive Test Suite Runner"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
VERBOSE="-v"
COVERAGE=false
PARALLEL=false
CATEGORY="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -u|--unit)
            CATEGORY="unit"
            shift
            ;;
        -i|--integration)
            CATEGORY="integration"
            shift
            ;;
        -e|--e2e)
            CATEGORY="e2e"
            shift
            ;;
        -x|--stop-on-failure)
            STOP_FLAG="-x"
            shift
            ;;
        -q|--quiet)
            VERBOSE=""
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage          Generate coverage report"
            echo "  -p, --parallel          Run tests in parallel"
            echo "  -u, --unit              Run unit tests only"
            echo "  -i, --integration       Run integration tests only"
            echo "  -e, --e2e               Run end-to-end tests only"
            echo "  -x, --stop-on-failure   Stop on first failure"
            echo "  -q, --quiet             Reduce output verbosity"
            echo "  -h, --help              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check pytest is installed
echo -e "${BLUE}Checking dependencies...${NC}"
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing test dependencies...${NC}"
    pip install -q -r tests/requirements-test.txt
fi
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Build pytest command
PYTEST_CMD="pytest"

case $CATEGORY in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        echo -e "${BLUE}Running Unit Tests${NC}"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        echo -e "${BLUE}Running Integration Tests${NC}"
        ;;
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/integration/test_e2e_workflows.py"
        echo -e "${BLUE}Running End-to-End Tests${NC}"
        ;;
    *)
        PYTEST_CMD="$PYTEST_CMD tests/"
        echo -e "${BLUE}Running All Tests${NC}"
        ;;
esac

# Add verbosity
[ -n "$VERBOSE" ] && PYTEST_CMD="$PYTEST_CMD $VERBOSE"

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    echo -e "${YELLOW}Using parallel execution${NC}"
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add stop on failure
[ -n "$STOP_FLAG" ] && PYTEST_CMD="$PYTEST_CMD $STOP_FLAG"

# Add coverage
if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Generating coverage report${NC}"
    PYTEST_CMD="$PYTEST_CMD --cov=lib --cov=use_cases --cov=app"
    PYTEST_CMD="$PYTEST_CMD --cov-report=html --cov-report=term-missing"
fi

echo ""
echo -e "${BLUE}Running command:${NC}"
echo "$PYTEST_CMD"
echo ""

# Run tests
$PYTEST_CMD

# Print summary
echo ""
echo -e "${GREEN}=================================================="
echo "Test execution completed!"
echo "==================================================${NC}"

# Print coverage info if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo -e "${BLUE}Coverage report generated:${NC}"
    echo "  View details at: htmlcov/index.html"
    echo ""
    if [ -f "htmlcov/index.html" ]; then
        echo -e "${YELLOW}Opening coverage report in browser...${NC}"
        # Try to open in browser (works on macOS and Linux)
        if command -v open &> /dev/null; then
            open htmlcov/index.html
        elif command -v xdg-open &> /dev/null; then
            xdg-open htmlcov/index.html
        fi
    fi
fi

echo ""
echo -e "${GREEN}✓ Done!${NC}"
