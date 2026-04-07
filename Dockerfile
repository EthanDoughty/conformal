# Build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy solution and project files first (layer caching)
COPY Conformal.sln .
COPY src/core/ConformalCore.fsproj src/core/
COPY src/shared/ConformalShared.fsproj src/shared/
COPY src/analyzer/ConformalAnalyzer.fsproj src/analyzer/
COPY src/migrate/ConformalMigrate.fsproj src/migrate/

# Restore dependencies
RUN dotnet restore Conformal.sln

# Copy source files
COPY src/core/ src/core/
COPY src/shared/ src/shared/
COPY src/analyzer/ src/analyzer/
COPY src/migrate/ src/migrate/

# Build release (skip AOT; container uses managed runtime)
RUN dotnet publish src/analyzer/ConformalAnalyzer.fsproj -c Release -o /app -p:PublishAot=false

# Runtime
FROM mcr.microsoft.com/dotnet/runtime:8.0 AS runtime
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "conformal.dll"]

# Test (runtime + test files)
FROM runtime AS test
COPY tests/ /app/tests/
ENTRYPOINT ["dotnet", "conformal.dll"]
