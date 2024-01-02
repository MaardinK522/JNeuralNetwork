package com.mkproductions.jnn.entity;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class CSVBufferedReader {
    private final String filePath;

    public CSVBufferedReader(String filePath) {
        this.filePath = filePath;
    }

    public int getRowCount() {
        int rowCount;
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(this.filePath))) {
            rowCount = bufferedReader.lines().toList().size();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return rowCount - 1;
    }

    public List<String> getLabels() {
        List<String> csvLabels = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(this.filePath))) {
            String row;
            row = bufferedReader.readLine();
            String[] labels = row.split(",");
            csvLabels.addAll(Arrays.asList(labels));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return csvLabels;
    }

    public List<Double> getColumn(String columnName) {
        List<Double> columnEntries = new ArrayList<>();
        int labelIndex = 0;
        for (int a = 0; a < this.getLabels().size(); a++) {
            if (Objects.equals(this.getLabels().get(a), columnName))
                break;
            labelIndex++;
        }
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(this.filePath))) {
            bufferedReader.readLine();
            String row;
            while ((row = bufferedReader.readLine()) != null) {
                String[] column = row.split(",");
                columnEntries.add(Double.valueOf(column[labelIndex]));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return columnEntries;
    }

    public List<List<Double>> getTable() {
        List<List<Double>> csvTable = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(this.filePath))) {
            bufferedReader.readLine();
            String rawRow;
            while ((rawRow = bufferedReader.readLine()) != null) {
                String[] column = rawRow.split(",");
                List<String> stringRow = Arrays.stream(column).toList();
                List<Double> doubleRow = new ArrayList<>();
                for (int a = 0; a < stringRow.size(); a++) {
                    if (a == 0) continue;
                    doubleRow.add(Double.parseDouble(stringRow.get(a)));
                }
                csvTable.add(doubleRow);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return csvTable;
    }
}
