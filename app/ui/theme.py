from __future__ import annotations

APP_STYLESHEET = """
QWidget {
    background: #f6efe6;
    color: #34281e;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10.5pt;
}
QMainWindow {
    background: #f2e6d8;
}
QLabel#headline {
    font-size: 20pt;
    font-weight: 700;
    color: #402d20;
}
QLabel#subtle {
    color: #7a6556;
}
QWidget#Card, QFrame#Card {
    background: #fffaf5;
    border: 1px solid #ead9c8;
    border-radius: 18px;
}
QWidget#AccentCard, QFrame#AccentCard {
    background: #fff3df;
    border: 1px solid #efcc9f;
    border-radius: 18px;
}
QPushButton {
    background: #bc6c25;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 7px 11px;
    font-weight: 600;
}
QPushButton:hover {
    background: #a95e1f;
}
QPushButton:pressed {
    background: #8e4e18;
}
QPushButton:disabled {
    background: #d8c8b7;
    color: #8a7a6a;
}
QPushButton[secondary="true"] {
    background: #f2dfc7;
    color: #5a4130;
    border: 1px solid #e3c49b;
}
QPushButton[secondary="true"]:hover {
    background: #ebd3b1;
}
QLineEdit, QTextEdit, QListWidget {
    background: #fffdf9;
    border: 1px solid #e7d6c2;
    border-radius: 14px;
    padding: 8px 10px;
}
QTextEdit#StepInstruction {
    background: #fffaf6;
    border: 1px solid #ead7c2;
    border-radius: 14px;
    padding: 12px;
    font-size: 11pt;
}
QSplitter::handle {
    background: #ead9c8;
    border-radius: 4px;
}
QCheckBox {
    spacing: 10px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #d1b89c;
    border-radius: 9px;
    background: #fffaf3;
}
QCheckBox::indicator:checked {
    border: 1px solid #bc6c25;
    border-radius: 9px;
    background: #bc6c25;
}
QScrollArea {
    border: none;
    background: transparent;
}
QFrame#StatusPill {
    border-radius: 12px;
    background: #f1e2d2;
    border: 1px solid #e0c3a0;
}
QFrame#StatusPill[active="true"] {
    background: #dff3df;
    border: 1px solid #88bb88;
}
QFrame#StatusPill[warning="true"] {
    background: #fff2cc;
    border: 1px solid #d6b85d;
}
QFrame#StatusBadge {
    background: #fff8ef;
    border: 1px solid #ead9c8;
    border-radius: 14px;
}
QLabel#PillTitle {
    font-size: 9pt;
    color: #775a44;
}
QLabel#PillValue {
    font-size: 11pt;
    font-weight: 700;
}
QLabel#TimerHeroLabel {
    font-size: 9.5pt;
    font-weight: 700;
    color: #8a4d16;
}
QLabel#TimerHeroTime {
    font-size: 28pt;
    font-weight: 800;
    color: #9a8b7d;
    padding: 0;
}
QLabel#TimerHeroTime[active="true"] {
    color: #b83b2d;
}
QLabel#TimerList {
    color: #6a5444;
    font-size: 8.8pt;
}
QFrame#RecipeOption[selected="true"] {
    background: #fff1dc;
    border: 2px solid #bc6c25;
    border-radius: 18px;
}
QFrame#RecipeOption[selected="false"] {
    background: #fffaf5;
    border: 1px solid #ead9c8;
    border-radius: 18px;
}
"""
