import sys
import matplotlib
matplotlib.use("Agg")
import schemdraw
import schemdraw.elements as elm

OUT = sys.argv[1]
schemdraw.use("matplotlib")

RED, BLUE, GREEN, GREY = "#B71C1C", "#1565C0", "#2E7D32", "#6B7280"


def pin(name, side, num=None):
    return elm.IcPin(name=name, side=side, pin=str(num) if num else "", anchorname=name)


with schemdraw.Drawing(show=False) as d:
    d.config(fontsize=13, unit=2.5, lw=2.4)

    # ------------------------------------------------------------ battery stack BT1..BT3 (bottom -> top, + at top)
    n0 = (0, 0)
    d += (bt1 := elm.BatteryCell().up().reverse().at(n0).length(3).label("BT1\n18650", loc="bottom", ofst=0.6))
    d += (bt2 := elm.BatteryCell().up().reverse().at(bt1.end).length(3).label("BT2\n18650", loc="bottom", ofst=0.6))
    d += (bt3 := elm.BatteryCell().up().reverse().at(bt2.end).length(3).label("BT3\n18650", loc="bottom", ofst=0.6))
    n1, n2, n3 = bt1.end, bt2.end, bt3.end
    for p in (n0, n1, n2, n3):
        d += elm.Dot().at(p)

    # ------------------------------------------------------------ U1: 3S BMS
    bms = elm.Ic(pins=[pin("B−\n(0V)", "left"), pin("B1\n(4.2V)", "left"), pin("B2\n(8.4V)", "left"),
                       pin("B+\n(12.6V)", "left"), elm.IcPin(name="P−", side="right", slot="1/4", anchorname="P−")],
                 pinspacing=3, edgepadH=0.8, w=3.2, leadlen=1, lsize=10, label="U1\n3S BMS\n20 A")
    d += bms.right().anchor("B−\n(0V)").at((4.5, 0))
    bx = bms.absanchors["B+\n(12.6V)"]
    d += elm.Wire("-").at(n0).to(bms.absanchors["B−\n(0V)"])
    d += elm.Wire("-").at(n1).to(bms.absanchors["B1\n(4.2V)"]).color("#E67E22")
    d += elm.Wire("-").at(n2).to(bms.absanchors["B2\n(8.4V)"]).color("#E67E22")
    d += elm.Wire("-").at(n3).to(bms.absanchors["B+\n(12.6V)"])

    # ------------------------------------------------------------ PACK+ rail: BT3+ -> F1 -> SW1A -> R1 -> R2
    top = 13.5
    d += elm.Line().at(n3).up().toy(top).color(RED)
    d += elm.Line().right().length(4).color(RED)
    d += elm.Dot().color(RED).label("PACK+", loc="top", color=RED)
    d += elm.Line().right().length(3.5).color(RED)
    d += (sw := elm.SwitchSpdt2().right().anchor("a").color(RED)
          .label("SW1  SPDT (3-pin), ≥ 3 A", loc="top", ofst=1.2))
    d += elm.Label().at((sw.absanchors["a"][0] - 0.2, sw.absanchors["a"][1] - 0.9)).label("C", fontsize=9)
    d += elm.Label().at((sw.absanchors["c"][0] + 1.6, sw.absanchors["c"][1])).label("3rd pin: unused", fontsize=9, color=GREY)
    # the load column must sit RIGHT of the shunt so its return lands on the S2 (load) side
    d += elm.Line().at(sw.absanchors["b"]).right().tox(bms.absanchors["P−"][0] + 8.0).color(RED)
    lt = d.here
    d += elm.Dot()
    d += elm.Resistor().down().length(3.2).label("R1\n3.3 Ω 25 W", loc="bottom")
    d += elm.Resistor().down().length(3.2).label("R2\n3.3 Ω 25 W", loc="bottom")
    lb = d.here
    d += elm.Dot()

    # ------------------------------------------------------------ return: R2 -> RS1 (shunt) -> BMS P-
    bot = -3.5
    pmx = bms.absanchors["P−"][0]
    s1 = (pmx + 3.6, bot)
    s2 = (s1[0] + 2.2, bot)
    d += elm.Line().at(lb).down().toy(bot)
    d += elm.Line().left().tox(s2[0])
    d += elm.Dot().at(s2).color(BLUE)
    d += elm.Resistor().at(s2).left().length(2.2).label("RS1 shunt\n200 A / 75 mV", loc="top")
    d += elm.Dot().at(s1).color(BLUE)
    d += elm.Line().at(s1).left().tox(pmx)
    d += elm.Line().up().toy(bms.absanchors["P−"][1])
    d += elm.Label().at((pmx + 0.3, bot - 0.9)).label("PACK−", fontsize=11)

    # ------------------------------------------------------------ U2: PZEM-017 (sense pins under the shunt)
    pz = elm.Ic(pins=[pin("V+", "top"), pin("V−", "top"), pin("S1", "top"), pin("S2", "top"),
                      pin("5V", "right"), pin("A", "right"), pin("B", "right"), pin("GND", "right")],
                pinspacing=2.2, edgepadW=1.0, edgepadH=0.9, w=9, leadlen=1, lsize=10,
                label="U2  PZEM-017\n(measuring side isolated\nfrom RS485 side)")
    d += pz.right().anchor("S1").at((s1[0], bot - 3.2))
    d += elm.Line().at(pz.absanchors["S1"]).to(s1).color(BLUE)
    d += elm.Line().at(pz.absanchors["S2"]).to(s2).color(BLUE)
    vm = pz.absanchors["V−"]
    d += elm.Line().at(vm).to((vm[0], bot)).color(BLUE)
    d += elm.Dot().at((vm[0], bot)).color(BLUE)
    # V+ -> PACK+ via net label
    d += elm.Line().at(pz.absanchors["V+"]).up().length(0.9).color(BLUE)
    d += elm.Dot(open=True).label("PACK+", loc="left", color=RED)

    # ------------------------------------------------------------ U3: MAX485 (bus side left, logic side right)
    mx = elm.Ic(pins=[pin("GND", "left", 5), pin("A", "left", 6), pin("B", "left", 7), pin("VCC", "left", 8),
                      pin("DI", "right", 4), pin("DE", "right", 3), pin("RE", "right", 2), pin("RO", "right", 1)],
                pinspacing=1.6, edgepadH=0.8, w=3.2, leadlen=1, lsize=10, label="U3\nMAX485")
    d += mx.right().anchor("GND").at((23.5, -15.4))
    _p, _q = pz.absanchors["A"], mx.absanchors["A"]
    d += elm.Line().at(_p).to((_p[0] + 2.0, _p[1])).color(GREEN)
    d += elm.Line().at((_p[0] + 2.0, _p[1])).to((_p[0] + 2.0, _q[1])).color(GREEN)
    d += elm.Line().at((_p[0] + 2.0, _q[1])).to(_q).color(GREEN)
    _p, _q = pz.absanchors["B"], mx.absanchors["B"]
    d += elm.Line().at(_p).to((_p[0] + 3.2, _p[1])).color(GREEN)
    d += elm.Line().at((_p[0] + 3.2, _p[1])).to((_p[0] + 3.2, _q[1])).color(GREEN)
    d += elm.Line().at((_p[0] + 3.2, _q[1])).to(_q).color(GREEN)
    # PZEM 5V / GND (RS485 side)
    d += elm.Line().at(pz.absanchors["5V"]).right().length(1.2)
    d += elm.Vdd().right().label("+5V")
    d += elm.Line().at(pz.absanchors["GND"]).right().length(1.2)
    d += elm.Ground().right()
    d += elm.Line().at(mx.absanchors["VCC"]).left().length(1.4)
    d += elm.Vdd().right().label("+5V")
    d += elm.Line().at(mx.absanchors["GND"]).left().length(1.4)
    d += elm.Ground().right()

    # ------------------------------------------------------------ U4: ESP32
    esp = elm.Ic(pins=[pin("GND", "left"), pin("GPIO27", "left"), pin("GPIO4", "left"), pin("GPIO17", "left"),
                       pin("GPIO16", "left"), pin("5V (VIN)", "left")],
                 pinspacing=1.6, edgepadH=1.0, w=3.6, leadlen=1, lsize=10)
    d += esp.right().anchor("GPIO17").at((36.5, mx.absanchors["DI"][1]))
    _t = esp.absanchors["5V (VIN)"]
    d += elm.Label().at((_t[0] + 2.8, _t[1] + 2.4)).label("U4  ESP32 Dev Module\n(USB → laptop)", fontsize=11)
    d += elm.Wire("-").at(mx.absanchors["DI"]).to(esp.absanchors["GPIO17"]).color(GREEN)
    # DE + RE joined -> GPIO4
    d += elm.Wire("-").at(mx.absanchors["DE"]).to((30.5, mx.absanchors["DE"][1])).color(GREEN)
    d += elm.Wire("-").at(mx.absanchors["RE"]).to((30.5, mx.absanchors["RE"][1])).color(GREEN)
    d += elm.Line().at((30.5, mx.absanchors["RE"][1])).to((30.5, mx.absanchors["DE"][1])).color(GREEN)
    d += elm.Dot().at((30.5, mx.absanchors["DE"][1])).color(GREEN)
    d += elm.Wire("|-").at((30.5, mx.absanchors["DE"][1])).to(esp.absanchors["GPIO4"]).color(GREEN)
    # RO -> R4 1k -> GPIO16 ; R5 2k -> GND
    d += elm.Line().at(mx.absanchors["RO"]).right().length(0.6).color(GREEN)
    d += elm.Resistor().right().length(2.6).label("R4 1 kΩ", loc="top")
    ro_node = d.here
    d += elm.Dot().color(GREEN)
    d += elm.Wire("-|").at(ro_node).to(esp.absanchors["GPIO16"]).color(GREEN)
    d += elm.Resistor().at(ro_node).up().length(2.4).label("R5 2 kΩ", loc="bottom")
    d += elm.Line().right().length(1.0)
    d += elm.Ground().right()
    # 5V and GND of ESP32
    d += elm.Line().at(esp.absanchors["5V (VIN)"]).left().length(1.4)
    d += elm.Vdd().right().label("+5V")
    # GND pin -> ground; SW1B (second pole of SW1) between GPIO27 and GND
    g = esp.absanchors["GND"]
    g27 = esp.absanchors["GPIO27"]
    d += elm.Line().at(g).left().length(2.2)
    gn = d.here
    d += elm.Dot()
    d += elm.Ground().right()
    d += elm.Label().at((g27[0] - 0.3, g27[1])).label("NC", fontsize=10, halign="right", color=GREY)
    # ------------------------------------------------------------ notes
    d += elm.Label().at((-1, -20)).label(
        "NOTES\n"
        "• SW1 is a 3-pin SPDT switch: middle pin (C) to PACK+, the pin that closes when ON to R1; the third pin is left unused.\n"
        "• Load: R1 + R2 = 2 × 3.3 Ω 25 W in SERIES = 6.6 Ω (≈ 1.9 A falling to 1.2 A). Resistors get very hot: mount on tile/metal.\n"
        "• GPIO27 is not connected (NC): firmware DISCHARGE_ONLY_RIG = true, because the cells are charged off-board on TP4056 modules.\n"
        "• PACK+/PACK− (battery side) and GND/+5V (ESP32 side) are separate: the PZEM-017 isolates its measuring side from its RS485 side.\n"
        "• PACK− is the BMS P− output (never B−). PZEM-017 is powered by the pack (≥ 7 V): no USB supply. BMS pads may read 0V/4.2V/8.4V/12.6V.\n"
        "• Firmware: CELLS_IN_SERIES = 3 (voltage, power and energy are sent per cell).  A dot marks a connection; crossings without a dot are not connected.",
        fontsize=10, halign="left", valign="top")

    d.save(OUT, dpi=170)
print("saved", OUT)
