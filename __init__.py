from __future__ import print_function

import struct, ctypes

from collections import defaultdict
from binaryninja import (
    Architecture, RegisterInfo, InstructionInfo,
    InstructionTextToken, InstructionTextTokenType,
    BranchType,
    LowLevelILOperation, LLIL_TEMP,
    LowLevelILLabel,
    FlagRole,
    LowLevelILFlagCondition,
    log_error,
    CallingConvention)

def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)

REGISTER_REGISTER_MODE = 0         # r1, r2
SINGLE_REGISTER_MODE = 1           # r1
INDIRECT_REGISTER_MODE = 2         # [r1]
INDIRECT_REGISTER_OFFSET_MODE = 3  # 
ELEMENT_OFFSET_LOAD_MODE = 4       # sld.bu -25 [ep], r2
ELEMENT_OFFSET_STORE_MODE = 5      # sst.b r2, -25 [ep]
IMMEDIATE_REGISTER_MODE = 6        # mov 5, r2
IMMEDIATE_MODE = 7                 # callt 25
OFFSET_MODE = 8                    # jr 123
OFFSET_REGISTER_MODE = 9           # 

Registers = [
    'r0',
    'r1',
    'r2',
    'sp',
    'r4',
    'r5',
    'r6',
    'r7',
    'r8',
    'r9',
    'r10',
    'r11',
    'r12',
    'r13',
    'r14',
    'r15',
    'r16',
    'r17',
    'r18',
    'r19',
    'r20',
    'r21',
    'r22',
    'r23',
    'r24',
    'r25',
    'r26',
    'r27',
    'r28',
    'r29',
    'ep',
    'lp',
    'pc',
]

NoRegisterInstructions = {
    0x0: 'nop',
    0x1d: 'synce',
    0x1e: 'syncm',
    0x1f: 'syncp',
    0x40: 'rie',
}

ThirtyTwoBitNoRegisterInstructions = {
    0x07e00120: 'halt',
    0x07e00140: 'reti',
    0x07e00144: 'ctret',
    0x07e00148: 'eiret',
    0x07e0014a: 'feret',
    0x07e00160: 'di',
    0x87e00160: 'ei'
}

OneRegInstructions = [
    'switch', 'zxb', 'sxb', 'zxh', 'sxh'
]

TwoRegInstructions = [
    'mov', 'not', 'divh', 'satsubr', 'satsub', 'satadd', 'mulh', 'or', 'xor',
    'and', 'tst', 'subr', 'sub', 'add', 'cmp',
]

RegImmediateInstructions = [
    'mov', 'satadd', 'add', 'cmp', 'shr', 'sar', 'shl', 'mulh'
]

ImmediateRegRegInstructions = [
    'addi', 'movea', 'movhi', 'satsubi', 'ori', 'xori', 'andi', 'mulhi'
]

LoadInstructions = [
    'ld.b', 'ld.bu', 'ld.h', 'ld.hu', 'ld.w',
    'sld.b', 'sld.bu', 'sld.h', 'sld.hu', 'sld.w'
]

StoreInstructions = [
    'st.b', 'st.h', 'st.w',
    'sst.b', 'sst.h', 'sst.w'
]

# TODO: instructions with conditions: ADF, SBF, SETF, B, CMOV, SASF

TypeOneRegRegInstructions = {
    0x00: lambda reg1, reg2: ('mov', reg1, reg2),
    0x01: lambda reg1, reg2: ('not', reg1, reg2),
    0x02: lambda reg1, reg2: ('switch', None, reg1) if reg2 == 0 else ('fetrap', None, reg2 & 0xf) if reg1 == 0 else ('divh', reg1, reg2),
    0x04: lambda reg1, reg2: ('satsubr', reg1, reg2) if reg2 != 0 else ('zxb', None, reg1),
    0x05: lambda reg1, reg2: ('satsub', reg1, reg2) if reg2 != 0 else ('sxb', None, reg1),
    0x06: lambda reg1, reg2: ('satadd', reg1, reg2) if reg2 != 0 else ('zxh', None, reg1),
    0x07: lambda reg1, reg2: ('mulh', reg1, reg2) if reg2 != 0 else ('sxh', None, reg1),
    0x08: lambda reg1, reg2: ('or', reg1, reg2),
    0x09: lambda reg1, reg2: ('xor', reg1, reg2),
    0x0a: lambda reg1, reg2: ('and', reg1, reg2),
    0x0b: lambda reg1, reg2: ('tst', reg1, reg2),
    0x0c: lambda reg1, reg2: ('subr', reg1, reg2),
    0x0d: lambda reg1, reg2: ('sub', reg1, reg2),
    0x0e: lambda reg1, reg2: ('add', reg1, reg2),
    0x0f: lambda reg1, reg2: ('cmp', reg1, reg2),
}

TypeOneImmediateRegInstructions = {
    0x03: lambda reg1, reg2: ('jmp', None, reg1) if reg2 == 0 else ('sld.hu', (reg1 & 0xf) << 1, reg2) if reg1 & 0x10 == 0x10 else ('sld.bu', reg1 & 0xf, reg2),
    0x10: lambda reg1, reg2: ('mov', sign_extend(reg1, 5), reg2),
    0x11: lambda reg1, reg2: ('satadd', reg1, reg2),
    0x12: lambda reg1, reg2: ('add', sign_extend(reg1, 5), reg2),
    0x13: lambda reg1, reg2: ('cmp', sign_extend(reg1, 5), reg2),
    0x14: lambda reg1, reg2: ('callt', reg2 << 1, None) if reg1 == 0 else ('shr', reg1, reg2),
    0x15: lambda reg1, reg2: ('callt', (reg2 | 0x20) << 1, None) if reg1 == 0 else ('sar', reg1, reg2),
    0x16: lambda reg1, reg2: ('shl', reg1, reg2),
}

TypeFourShortLoadStoreInstructions = {
    0x6: lambda reg2, disp: ('sld.b', None, reg2, disp),
    0x7: lambda reg2, disp: ('sst.b', reg2, None, disp),
    0x8: lambda reg2, disp: ('sld.h', None, reg2, disp << 1),
    0x9: lambda reg2, disp: ('sst.h', reg2, None, disp << 1),
    0xa: lambda reg2, disp: ('sld.w', None, reg2, disp << 1) if disp & 0x1 == 0 else ('sst.w', reg2, None, (disp & 0x7e) << 1),
}

BitManipulationInstructions = [
    lambda reg1, bitnum, disp: ('set1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('not1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('clr1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('tst1', 4, reg1, bitnum, sign_extend(disp, 16)),
]

class Register_List_Bits (ctypes.LittleEndianStructure):
    _fields_ = [
        ('r31', ctypes.c_uint16, 1),
        ('r29', ctypes.c_uint16, 1),
        ('r28', ctypes.c_uint16, 1),
        ('r23', ctypes.c_uint16, 1),
        ('r22', ctypes.c_uint16, 1),
        ('r21', ctypes.c_uint16, 1),
        ('r20', ctypes.c_uint16, 1),
        ('r27', ctypes.c_uint16, 1),
        ('r26', ctypes.c_uint16, 1),
        ('r25', ctypes.c_uint16, 1),
        ('r24', ctypes.c_uint16, 1),
        ('r30', ctypes.c_uint16, 1),
    ]
class Register_List (ctypes.Union):
    _anonymous_ = ('bit',)
    _fields_ = [
        ('bit', Register_List_Bits),
        ('asWord', ctypes.c_uint16)
    ]

def decode_dispose(i1, i2):
    immed = ((i1 >> 1) & 0x1f) << 2
    reg1 = i2 & 0x1f
    reg_list = Register_List()
    reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)

    return 'dispose', 4, reg1 if reg1 != 0 else None, reg_list, immed

def decode_3c_and_3d(data):
    i1 = struct.unpack('<H', data[0:2])[0]
    i2 = struct.unpack('<H', data[2:4])[0]
    reg1 = i1 & 0x1f
    reg2 = i1 >> 11

    if i2 & 0x1 == 0 and reg2 != 0:
        return 'jarl', 4, None, reg2, sign_extend(((i1 & 0x3f) << 16) | i2, 22)
    if i2 & 0x1 == 0 and reg2 == 0:
        return 'jr', 4, None, None, sign_extend(((i1 & 0x3f) << 16) | i2, 22)
    if reg2 != 0:
        reg1 = (i1 & 0x1f)
        low_bit = (i1 >> 5) & 0x1
        return 'ld.bu', 4, reg1, reg2, sign_extend((i2 & 0xfffe) | low_bit, 16)

    subop = i2 & 0x1f

    if subop == 1:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        return 'prepare', 4, None, reg_list, immed
    if subop == 3:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        return 'prepare', 4, Registers.index('sp'), reg_list, immed

    i3 = struct.unpack('<H', data[4:6])[0]

    if subop == 0xb or subop == 0x13:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        return 'prepare', 6, sign_extend(i3, 16) if subop == 0xb else i3 << 16, reg_list, immed
    if subop == 0x1b:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        i4 = struct.unpack('<H', data[6:8])[0]
        return 'prepare', 8, i4 << 16 | i3, reg_list, immed

    composite_opcode = (i1 & 0x20) | (i2 & 0xf)
    reg3 = (i2 >> 11) & 0x1f
    disp23 = ((i2 >> 4) & 0x7f) | (i3 << 7)
    if composite_opcode in LongLoadStoreInstructions:
        return LongLoadStoreInstructions[composite_opcode](reg1, reg3, disp23)
    return None, None, None, None, None

LongLoadStoreInstructions = {
    0x05: lambda reg1, reg3, disp23: ('ld.b', 6, reg1, reg3, sign_extend(disp23, 23)),
    0x07: lambda reg1, reg3, disp23: ('ld.h', 6, reg1, reg3, sign_extend(disp23, 23)),
    0x09: lambda reg1, reg3, disp23: ('ld.w', 6, reg1, reg3, sign_extend(disp23, 23)),
    0x0d: lambda reg1, reg3, disp23: ('st.b', 6, reg3, reg1, sign_extend(disp23, 23)),
    0x0f: lambda reg1, reg3, disp23: ('st.w', 6, reg3, reg1, sign_extend(disp23, 23)),
    0x25: lambda reg1, reg3, disp23: ('ld.bu', 6, reg1, reg3, sign_extend(disp23, 23)),
    0x27: lambda reg1, reg3, disp23: ('ld.hu', 6, reg1, reg3, sign_extend(disp23, 23)),
    0x2d: lambda reg3, reg1, disp23: ('st.h', 6, reg1, reg3, sign_extend(disp23, 23)),
}

SimpleThirtyTwoBitInstructions = {
    0x30: lambda reg1, reg2, _, i2, _2: ('addi', 4, reg1, reg2, sign_extend(i2, 16)),
    0x31: lambda reg1, reg2, _, i2, i3: ('movea', 4, reg1, reg2, sign_extend(i2, 16)) if reg2 != 0 else ('mov', 6, None, reg1, (i3 << 16) | i2),
    0x32: lambda reg1, reg2, i1, i2, i3: ('movhi', 4, reg1, reg2, i2) if reg2 != 0 else decode_dispose(i1, i2),
    0x33: lambda reg1, reg2, i1, i2, i3: ('satsubi', 4, reg1, reg2, i2 << 16) if reg2 != 0 else decode_dispose(i1, i2),
    0x34: lambda reg1, reg2, _, i2, _2: ('ori', 4, reg1, reg2, i2),
    0x35: lambda reg1, reg2, _, i2, _2: ('xori', 4, reg1, reg2, i2),
    0x36: lambda reg1, reg2, _, i2, _2: ('andi', 4, reg1, reg2, i2),
    0x37: lambda reg1, reg2, _, i2, i3: ('mulhi', 4, reg1, reg2, sign_extend(i2, 16)) if reg2 != 0 else ('jmp', 6, None, reg1, (i3 << 16) | i2),
    0x38: lambda reg1, reg2, _, i2, _2: ('ld.b', 4, reg1, reg2, sign_extend(i2, 16)),
    0x39: lambda reg1, reg2, _, i2, _2: ('ld.h', 4, reg1, reg2, sign_extend(i2, 16)) if i2 & 0x1 == 0 else ('ld.w', 4, reg1, reg2, sign_extend(i2 & 0xfffe, 16)),
    0x3a: lambda reg1, reg2, _, i2, _2: ('st.b', 4, reg2, reg1, sign_extend(i2, 16)),
    0x3b: lambda reg1, reg2, _, i2, _2: ('st.h', 4, reg2, reg1, sign_extend(i2, 16)) if i2 & 0x1 == 0 else ('st.w', 4, reg2, reg1, sign_extend(i2 & 0xfffe, 16)),
}

ExtendedInstructions = {
    0x01: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0: lambda reg1, reg2, _, _2: ('ldsr', 4, reg1, reg2, None, None, None, None)
    }),
    0x02: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0: lambda reg1, reg2, _, _2: ('stsr', 4, reg1, reg2, None, None, None, None)
    }),
    0x04: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, _2: ('shr', 4, reg1, reg2, None, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('shr', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x05: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, _2: ('sar', 4, reg1, reg2, None, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('sar', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x06: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, _2: ('shl', 4, reg1, reg2, None, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('shl', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x07: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, _2: ('set1', 4, reg1, reg2, None, None, None, None),
        0x2: lambda reg1, reg2, _, _2: ('not1', 4, reg1, reg2, None, None, None, None),
        0x4: lambda reg1, reg2, _, _2: ('clr1', 4, reg1, reg2, None, None, None, None),
        0x6: lambda reg1, reg2, _, _2: ('tst1', 4, reg1, reg2, None, None, None, None),
        0xe: lambda reg1, reg2, _, i2: ('caxi', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x0b: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, i1, i2: ('syscall', 4, None, None, None, None, ((i2 >> 11) & 0x7) | (i1 & 0x1f), None),
    }),
    0x10: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, _2: ('sasf', 4, None, reg2, None, None, None, reg1 & 0xf),
    }),
    0x11: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('mul', 4, reg1, reg2, i2 >> 11, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('mulu', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x12: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x4: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x8: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0xc: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x10: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x14: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x18: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x1c: lambda reg1, reg2, _, i2: ('mul', 4, None, reg2, i2 >> 11, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x2: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0x6: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0xa: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0xe: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0x12: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0x16: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0x1a: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
        0x1e: lambda reg1, reg2, _, i2: ('mulu', 4, None, reg2, i2 >> 11, None, reg1 | ((i2 & 0x3d) << 3), None),
    }),
    0x13: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('mul', 4, reg2, i2 >> 11, None, None, sign_extend(reg1 | ((i2 & 0x3f) << 3), 9), None),
        0x2: lambda reg1, reg2, _, i2: ('mulu', 4, reg2, i2 >> 11, None, None, reg1 | ((i2 & 0x3d) << 3), None),
    }),
    0x14: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('divh', 4, reg1, reg2, i2 >> 11, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('divhu', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x16: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('div', 4, reg1, reg2, i2 >> 11, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('divu', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x17: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x1c: lambda reg1, reg2, _, i2: ('divq', 4, reg1, reg2, i2 >> 11, None, None, None),
        0x1e: lambda reg1, reg2, _, i2: ('divqu', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x18: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('cmov', 4, reg2, i2 >> 11, None, None, reg1, sign_extend((i2 >> 1) & 0xf, 5))),
    0x19: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('cmov', 4, reg2, i2 >> 11, reg1, None, reg1, None)),
    0x1a: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('bsw', 4, reg2, i2 >> 11, None, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('bsh', 4, reg2, i2 >> 11, None, None, None, None),
        0x4: lambda reg1, reg2, _, i2: ('hsw', 4, reg2, i2 >> 11, None, None, None, None),
        0x6: lambda reg1, reg2, _, i2: ('hsh', 4, reg2, i2 >> 11, None, None, None, None),
    }),
    0x1b: defaultdict(lambda: lambda r1, r2, i1, i2: (None, None, None, None, None, None, None, None), {
        0x0: lambda reg1, reg2, _, i2: ('sch0r', 4, reg2, i2 >> 11, None, None, None, None),
        0x2: lambda reg1, reg2, _, i2: ('sch1r', 4, reg2, i2 >> 11, None, None, None, None),
        0x4: lambda reg1, reg2, _, i2: ('sch0l', 4, reg2, i2 >> 11, None, None, None, None),
        0x6: lambda reg1, reg2, _, i2: ('sch1l', 4, reg2, i2 >> 11, None, None, None, None),
    }),
    0x1c: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('sbf', 4, reg1, reg2, i2 >> 11, None, reg1, (i2 >> 1) & 0xf), {
        0x1a: lambda reg1, reg2, _, i2: ('satsub', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x1d: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('adf', 4, reg1, reg2, i2 >> 11, None, reg1, (i2 >> 1) & 0xf), {
        0x1a: lambda reg1, reg2, _, i2: ('satadd', 4, reg1, reg2, i2 >> 11, None, None, None),
    }),
    0x1e: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('mac', 4, reg1, reg2, i2 >> 11, i2 & 0x1f, None, None)),
    0x1f: defaultdict(lambda: lambda reg1, reg2, i1, i2: ('macu', 4, reg1, reg2, i2 >> 11, i2 & 0x1f, None, None)),
}

ConditionCode = [
    'v', 'l', 'z', 'nh',
    'n', 't', 'lt', 'le',
    'nv', 'nl', 'nz', 'h',
    'p', 'sa', 'ge', 'gt'
]

BranchConditionCode = [
    'bv', 'bl', 'be', 'bnh',
    'bn', 'br', 'blt', 'ble',
    'bnv', 'bnl', 'bne', 'bh',
    'bp', 'bsa', 'bge', 'bgt'
]

BranchConditionToILCondition = [
    LowLevelILFlagCondition.LLFC_O,
    LowLevelILFlagCondition.LLFC_ULT,
    LowLevelILFlagCondition.LLFC_E,
    LowLevelILFlagCondition.LLFC_ULE,
    LowLevelILFlagCondition.LLFC_NEG,
    None,
    LowLevelILFlagCondition.LLFC_SLT,
    LowLevelILFlagCondition.LLFC_SLE,
    LowLevelILFlagCondition.LLFC_NO,
    LowLevelILFlagCondition.LLFC_UGE,
    LowLevelILFlagCondition.LLFC_NE,
    LowLevelILFlagCondition.LLFC_UGT,
    LowLevelILFlagCondition.LLFC_POS,
    None, # Saturated
    LowLevelILFlagCondition.LLFC_SGE,
    LowLevelILFlagCondition.LLFC_SGT,
]

StorageSize = {'b': 1, 'h': 2, 'w': 4, 'bu': 1, 'hu': 2}

def to_il_src_reg(il, reg, width=4):
    return il.reg(width, Registers[reg]) if reg != 0 else il.const(0, 0)

def to_il_dst_reg(il, reg):
    return il.reg(4, Registers[reg]) if reg != 0 else il.undefined()

def to_il_set_reg(il, reg, val):
    return il.set_reg(4, reg, val) if reg != 'r0' else val

def cond_branch(il, cond, dest):
    t = None
    if il[dest].operation == LowLevelILOperation.LLIL_CONST:
        t = il.get_label_for_address(Architecture['V850'], il[dest].constant)
    if t is None:
        t = LowLevelILLabel()
        indirect = True
    else:
        indirect = False
    f = LowLevelILLabel()
    il.append(il.if_expr(cond, t, f))
    if indirect:
        il.mark_label(t)
        il.append(il.jump(dest))
    il.mark_label(f)
    return None

class V850(Architecture):
    name = 'V850'
    address_size = 4
    default_int_size = 4
    max_instr_length = 8
    stack_pointer = 'sp'
    link_reg = 'lp'

    regs = {
        'r0': RegisterInfo('r0', 4),
        'r1': RegisterInfo('r1', 4),
        'r2': RegisterInfo('r2', 4),
        'sp': RegisterInfo('sp', 4),
        'r4': RegisterInfo('r4', 4),
        'r5': RegisterInfo('r5', 4),
        'r6': RegisterInfo('r6', 4),
        'r7': RegisterInfo('r7', 4),
        'r8': RegisterInfo('r8', 4),
        'r9': RegisterInfo('r9', 4),
        'r10': RegisterInfo('r10', 4),
        'r11': RegisterInfo('r11', 4),
        'r12': RegisterInfo('r12', 4),
        'r13': RegisterInfo('r13', 4),
        'r14': RegisterInfo('r14', 4),
        'r15': RegisterInfo('r15', 4),
        'r16': RegisterInfo('r16', 4),
        'r17': RegisterInfo('r17', 4),
        'r18': RegisterInfo('r18', 4),
        'r19': RegisterInfo('r19', 4),
        'r20': RegisterInfo('r20', 4),
        'r21': RegisterInfo('r21', 4),
        'r22': RegisterInfo('r22', 4),
        'r23': RegisterInfo('r23', 4),
        'r24': RegisterInfo('r24', 4),
        'r25': RegisterInfo('r25', 4),
        'r26': RegisterInfo('r26', 4),
        'r27': RegisterInfo('r27', 4),
        'r28': RegisterInfo('r28', 4),
        'r29': RegisterInfo('r29', 4),
        'ep': RegisterInfo('ep', 4),
        'lp': RegisterInfo('lp', 4),
        'pc': RegisterInfo('pc', 4),
    }

    global_regs = ['pc']
    flags = ['sat', 'cy', 'ov', 's', 'z']
    flag_roles = {
        'sat': FlagRole.SpecialFlagRole,
        'cy': FlagRole.CarryFlagRole,
        'ov': FlagRole.OverflowFlagRole,
        's': FlagRole.NegativeSignFlagRole,
        'z': FlagRole.ZeroFlagRole,
    }
    flags_required_for_flag_condition = {
        LowLevelILFlagCondition.LLFC_O: ['ov'],
        LowLevelILFlagCondition.LLFC_NO: ['ov'],
        LowLevelILFlagCondition.LLFC_UGE: ['cy'],
        LowLevelILFlagCondition.LLFC_ULT: ['cy'],
        LowLevelILFlagCondition.LLFC_UGT: ['cy', 'z'],
        LowLevelILFlagCondition.LLFC_ULE: ['cy', 'z'],
        LowLevelILFlagCondition.LLFC_SGE: ['s', 'ov'],
        LowLevelILFlagCondition.LLFC_SGT: ['s', 'ov', 'z'],
        LowLevelILFlagCondition.LLFC_SLE: ['s', 'ov', 'z'],
        LowLevelILFlagCondition.LLFC_SLT: ['s', 'ov'],
        LowLevelILFlagCondition.LLFC_E: ['z'],
        LowLevelILFlagCondition.LLFC_NE: ['z'],
        LowLevelILFlagCondition.LLFC_NEG: ['s'],
        LowLevelILFlagCondition.LLFC_POS: ['s']
    }

    # The first flag write type is ignored currently.
    # See: https://github.com/Vector35/binaryninja-api/issues/513
    # flag_write_types = ['', '*', 'cnv', 'cnz']
    flag_write_types = ['', '*', 'ovsz']
    flags_written_by_flag_write_type = {
        "*": ["cy", "z", "ov", "s"],
        "ovsz": ["z", "ov", "s"],
    }

    def decode_instruction(self, data, addr):
        error_value = (None, None, None, None, None, None, None, None)
        if len(data) < 2:
            return error_value

        instruction = struct.unpack('<H', data[0:2])[0]
        # is this a zero-register instruction?
        if instruction in NoRegisterInstructions:
            return (NoRegisterInstructions[instruction], 2, None, None, None, None, None, None)

        opcode = (instruction >> 5) & 0x3f
        reg1 = (instruction & 0x1f)
        reg2 = (instruction >> 11) & 0x1f

        if opcode in TypeOneRegRegInstructions:
            instr, src, dst = TypeOneRegRegInstructions[opcode](reg1, reg2)
            if instr == 'mov' and reg2 is None:
                return error_value
            return (instr, 2, src, dst, None, None, None, None)

        if opcode in TypeOneImmediateRegInstructions:
            instr, immed, dst = TypeOneImmediateRegInstructions[opcode](reg1, reg2)
            return (instr, 2, None, dst, None, None, immed, None)

        # 0x17: mulh, jr, jarl
        if opcode == 0x17:
            if reg1 == reg2 == 0:
                return 'jr', 6, None, None, None, None, (struct.unpack('<H', data[4:6])[0] << 16) | struct.unpack('<H', data[2:4])[0], None
            if reg2 == 0:
                return 'jarl', 6, None, reg1, None, None, sign_extend((struct.unpack('<H', data[4:6])[0] << 16) | struct.unpack('<H', data[2:4])[0], 32), None
            else:
                return 'mulh', 2, None, reg2, None, None, sign_extend(reg1, 5), None

        # Handle short load store instructions
        if opcode >= 0x18 and opcode <= 0x2b:
            disp = instruction & 0x7f
            load_store_type = opcode >> 2
            instr, src, dst, immed = TypeFourShortLoadStoreInstructions[load_store_type](reg2, disp)
            return instr, 2, src, dst, None, None, immed, None

        # Handle branches
        if opcode >= 0x2c and opcode <= 0x2f:
            disp = (reg2 << 3) | ((instruction >> 4) & 0x7)
            return BranchConditionCode[instruction & 0xf], 2, None, None, None, None, sign_extend(disp << 1, 9), None

        instruction2 = struct.unpack('<H', data[2:4])[0]
        instruction_word = instruction << 16 | instruction2

        if instruction_word in ThirtyTwoBitNoRegisterInstructions:
            return ThirtyTwoBitNoRegisterInstructions[instruction_word], 4, None, None, None, None, None, None

        if opcode in SimpleThirtyTwoBitInstructions:
            if len(data) >= 6:
                instruction3 = struct.unpack('<H', data[4:6])[0]
            else:
                instruction3 = None
            instr, size, src, dst, immed = SimpleThirtyTwoBitInstructions[opcode](reg1, reg2, instruction, instruction2, instruction3)
            return instr, size, src, dst, None, None, immed, None

        if opcode == 0x3c or opcode == 0x3d:
            instr, size, src, dst, immed = decode_3c_and_3d(data)
            return instr, size, src, dst, None, None, immed, None

        if opcode == 0x3e:
            bitop = instruction >> 14
            bitnum = (instruction >> 11) & 0x7
            instr, size, src, dst, immed = BitManipulationInstructions[bitop](reg1, bitnum, instruction2)
            return instr, size, src, dst, None, None, immed, None

        if opcode == 0x3f and instruction2 & 0x1 == 0x1:
            disp = (instruction2 & 0xfffe)
            return 'ld.hu', 4, reg1, reg2, None, None, sign_extend(disp, 16), None

        if opcode == 0x3f and instruction2 == 0:
            if reg1 & 0x10 == 0:
                return 'setf', 4, reg1, reg2, None, None, None, None
            else:
                return 'rie', 4, reg2, reg1 & 0xf, None, None, None, None

        subop = (instruction2 >> 5) & 0x3f
        if opcode == 0x3f and subop in ExtendedInstructions:
            sub_subop = instruction2 & 0x1f
            instr, size, src, dst, r3, r4, immed, cond = ExtendedInstructions[subop][sub_subop](reg1, reg2, instruction, instruction2)
            return instr, size, src, dst, r3, r4, immed, cond

        # return instr, length, src_reg, dst_reg, reg3, reg4, immed, cond
        return error_value
    # def _get_link_register(self, ctxt):
    #     print(ctxt)
    #     return Registers.index('lp')

    def get_instruction_info(self, data, addr):
        instr, length, src_reg, dst_reg, reg3, reg4, immed, cond = self.decode_instruction(data, addr)
        if instr is None:
            return None

        result = InstructionInfo()
        result.length = length


        if instr[0] == 'b' and instr != 'bsw' and instr != 'bsh':
            branch_target = sign_extend(immed, 32) + addr
            if instr == 'br':
                result.add_branch(BranchType.UnconditionalBranch, branch_target)
            else:
                result.add_branch(BranchType.TrueBranch, branch_target)
                result.add_branch(BranchType.FalseBranch, addr + length)

        elif instr == 'jmp':
            # if dst_reg == Registers.index('lp'):
            #     result.add_branch(BranchType.FunctionReturn)
            # else:
                result.add_branch(BranchType.IndirectBranch)

        elif instr == 'jarl':
            branch_target = sign_extend(immed, 32) + addr
            result.add_branch(BranchType.CallDestination, branch_target)

        elif instr == 'jr':
            branch_target = sign_extend(immed, 32) + addr
            result.add_branch(BranchType.UnconditionalBranch, branch_target)

        elif instr == 'dispose' and dst_reg != None:
            result.add_branch(BranchType.FunctionReturn)

        elif instr == 'switch':
            result.add_branch(BranchType.IndirectBranch)
        elif instr in ['feret', 'eiret', 'ctret', 'reti']:
            result.add_branch(BranchType.FunctionReturn)

        return result
        # instr, _, _, _, _, _, length, src_value, _ = self.decode_instruction(data, addr)
        #
        # if instr is None:
        #     return None
        #
        # result = InstructionInfo()
        # result.length = length
        #
        # # Add branches
        # if instr in ['ret', 'reti']:
        #     result.add_branch(BranchType.FunctionReturn)
        # elif instr in ['jmp', 'br'] and src_value is not None:
        #     result.add_branch(BranchType.UnconditionalBranch, src_value)
        # elif instr in TYPE3_INSTRUCTIONS:
        #     result.add_branch(BranchType.TrueBranch, src_value)
        #     result.add_branch(BranchType.FalseBranch, addr + 2)
        # elif instr == 'call' and src_value is not None:
        #     result.add_branch(BranchType.CallDestination, src_value)
        #
        # return result

    def get_instruction_text(self, data, addr):
        instr, length, src_reg, dst_reg, reg3, reg4, immed, cond = self.decode_instruction(data, addr)
        if instr is None:
            return None

        instruction_text = instr
        tokens = [
            InstructionTextToken(InstructionTextTokenType.TextToken, '{:8s}'.format(instruction_text))
        ]

        # if instr in NoRegisterInstructions.values():
        #     return tokens, length
        if instr in OneRegInstructions and dst_reg is not None:
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in TwoRegInstructions and src_reg is not None and dst_reg is not None and immed is None:
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in RegImmediateInstructions and src_reg is None and dst_reg is not None and immed is not None:
            tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.TextToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in ImmediateRegRegInstructions and src_reg is not None and dst_reg is not None and immed is not None:
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.TextToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in LoadInstructions:
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '[')]
            if src_reg is None:
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, 'ep')]
            else:
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '], ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in StoreInstructions:
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '[')]
            if dst_reg is None:
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, 'ep')]
            else:
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ']')]

        if instr in ['mul', 'mulu']:
            if immed is not None:
                tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(immed), immed)]
            else:
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[reg3])]

        if instr == 'jmp':
            if immed is not None:
                tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '[')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ']')]

        if instr == 'jr':
            branch_target = immed + addr
            tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(branch_target), branch_target)]

        if instr == 'jarl':
            branch_target = immed + addr
            tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(branch_target), branch_target)]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr[0] == 'b' and instr != 'bsw' and instr != 'bsh':
            # TODO - consider what the best place to fix up immediates is
            branch_target = immed + addr
            tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(branch_target), branch_target)]

        if instr == 'prepare':
            registers = sorted(dst_reg.bit._fields_, key = lambda tup: tup[0])
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '{')]
            handled_a_register = False
            for r in registers:
                regname = r[0]
                printable_regname = regname
                if regname == 'r30': printable_regname = 'ep'
                if regname == 'r31': printable_regname = 'lp'
                if getattr(dst_reg.bit, regname):
                    handled_a_register = True
                    tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, printable_regname)]
                    tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            if handled_a_register: tokens.pop()
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '}, ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed), immed)]
            if src_reg != None:
                tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
                if length == 4:
                    tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
                else:
                    tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(src_reg))]

        if instr == 'dispose':
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed), immed)]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', {')]

            registers = sorted(dst_reg.bit._fields_, key = lambda tup: tup[0])
            handled_a_register = False
            for r in registers:
                regname = r[0]
                printable_regname = regname
                if regname == 'r30': printable_regname = 'ep'
                if regname == 'r31': printable_regname = 'lp'
                if getattr(dst_reg.bit, regname):
                    handled_a_register = True
                    tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, printable_regname)]
                    tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            if handled_a_register: tokens.pop()
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, '}')]
            if src_reg != None:
                tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
                tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]

        return tokens, length

    def get_instruction_low_level_il(self, data, addr, il):
        instr, length, src_reg, dst_reg, reg3, reg4, immed, cond = self.decode_instruction(data, addr)
        if instr is None:
            return None

        if instr in ['nop', 'synce', 'syncm', 'syncp', 'di', 'ei']:
            # TODO - something better we can emit for sync instructions?
            il.append(il.nop())
        elif instr in ['feret', 'eiret', 'ctret', 'reti']:
            # TODO - use real jump targets (if possible?)
            il.append(il.ret(il.reg(4, 'lp')))
        elif instr == 'jmp':
            il.append(il.jump(il.reg(4, Registers[dst_reg])))
            # il.append(il.ret(il.reg(4, Registers[dst_reg])))
        # elif instr == 'dispose' and src_reg != None:
        #     il.append(il.ret(to_il_src_reg(il, src_reg)))
        elif instr == 'dispose':
            if immed > 0:
                il.append(il.set_reg(4, 'sp', il.add(4, il.reg(4, 'sp'), il.const(4, immed))))
            registers = sorted(dst_reg.bit._fields_, key = lambda tup: tup[0], reverse = True)
            for r in registers:
                regname = r[0]
                printable_regname = regname
                if regname == 'r30': printable_regname = 'ep'
                if regname == 'r31': printable_regname = 'lp'
                if getattr(dst_reg.bit, regname):
                    il.append(il.set_reg(4, printable_regname, il.pop(4)))
            if src_reg != None:
                il.append(il.ret(to_il_src_reg(il, src_reg)))
        elif instr == 'prepare':
            registers = sorted(dst_reg.bit._fields_, key = lambda tup: tup[0])
            for r in registers:
                regname = r[0]
                printable_regname = regname
                if regname == 'r30': printable_regname = 'ep'
                if regname == 'r31': printable_regname = 'lp'
                if getattr(dst_reg.bit, regname):
                    il.append(il.push(4, il.reg(4, printable_regname)))
            if immed > 0:
                il.append(il.set_reg(4, 'sp', il.sub(4, il.reg(4, 'sp'), il.const(4, immed))))
            if src_reg != None:
                if length == 4:
                    il.append(il.set_reg(4, 'ep', to_il_src_reg(il, src_reg)))
                else:
                    il.append(il.set_reg(4, 'ep', il.const(4, src_reg)))
        elif instr == 'mov':
            if src_reg != None:
                il.append(to_il_set_reg(il, Registers[dst_reg], to_il_src_reg(il, src_reg)))
            else:
                il.append(to_il_set_reg(il, Registers[dst_reg], il.const(4, immed)))
        elif instr == 'movhi' and dst_reg != None:
            il.append(to_il_set_reg(il, Registers[dst_reg], il.add(4, to_il_src_reg(il, src_reg), il.shift_left(4, il.const(2, immed), il.const(1, 16)))))
        elif instr in StoreInstructions:
            if dst_reg == None:
                dst_reg = Registers.index('ep')
            il.append(il.store(StorageSize[instr.split('.')[1]],
                il.add(4, to_il_src_reg(il, dst_reg), il.const(4, immed)),
                to_il_src_reg(il, src_reg)))
        elif instr in LoadInstructions:
            if src_reg == None:
                src_reg = Registers.index('ep')
            if instr[-1] == 'u':
                extend = il.zero_extend
            else:
                extend = il.sign_extend
            il.append(to_il_set_reg(il, Registers[dst_reg],
                extend(4,
                    il.load(StorageSize[instr.split('.')[1]],
                    il.add(4, to_il_src_reg(il, src_reg), il.const(4, immed))))))
        elif instr == 'movea':
            il.append(to_il_set_reg(il, Registers[dst_reg],
                il.add(4, to_il_src_reg(il, src_reg), il.const(4, immed))))
        elif instr == 'jarl':
            branch_target = sign_extend(immed, 32) + addr
            # il.append(il.ret(il.const_pointer(4, branch_target)))
            il.append(il.call(il.const_pointer(4, branch_target)))
            # if dst_reg != Registers.index('lp'):
            #     il.append(il.set_reg(4, Registers[dst_reg], il.const(4, addr + length)))
            # il.append(il.set_reg(4, Registers[dst_reg], il.const(4, addr + length)))
            # il.append(il.jump(il.const_pointer(4, branch_target)))
            # TODO -consider this
            # if dst_reg == Registers.index('lp'):
            #     il.append(il.call(il.const_pointer(4, branch_target)))
            # else:
            #     il.append(il.set_reg(4, Registers[dst_reg], il.const(4, addr + length)))
            #     il.append(il.jump(il.const_pointer(4, branch_target)))
        elif instr == 'jr':
            branch_target = sign_extend(immed, 32) + addr
            il.append(il.jump(il.const(4, branch_target)))
        elif instr == 'add':
            src = to_il_src_reg(il, src_reg) if immed is None else il.const(4, immed)
            il.append(to_il_set_reg(il, Registers[dst_reg], il.add(4, src, to_il_src_reg(il, dst_reg), flags='*')))
        elif instr == 'addi':
            op1 = to_il_src_reg(il, src_reg)
            op2 = il.const(4, immed)
            # if dst_reg == 0:
            #     # TODO - this is a  massive hack to get flags set correctly while still allowing me to be lazy
            #     il.append(to_il_set_reg(il, Registers[dst_reg], il.sub(4, il.const(5, (-1 * immed) - 1), op1, flags='*')))
            # else:
            #     il.append(to_il_set_reg(il, Registers[dst_reg], il.add(4, op2, op1, flags='*')))
            il.append(to_il_set_reg(il, Registers[dst_reg], il.add(4, op2, op1, flags='*')))
        elif instr == 'sub':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.sub(4, to_il_src_reg(il, dst_reg), to_il_src_reg(il, src_reg), flags='*')))
        elif instr == 'subr':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.sub(4, to_il_src_reg(il, src_reg), to_il_src_reg(il, dst_reg), flags='*')))
        elif instr in ['mul', 'mulu']:
            operation = il.mult_double_prec_signed if instr == 'mul' else il.mult_double_prec_unsigned
            dst_hi = Registers[reg3]
            dst_lo = Registers[dst_reg]
            op1 = to_il_src_reg(il, src_reg) if immed is None else il.const(4, immed)
            mul_result = operation(8, to_il_src_reg(il, dst_reg), op1)
            il.append(to_il_set_reg(il, dst_hi, il.logical_shift_right(4, mul_result, il.const(4, 32))))
            if dst_hi != dst_lo:
                # we're not discarding the low 32 bits
                il.append(to_il_set_reg(il, dst_lo, mul_result))
        elif instr == 'mulh':
            src = to_il_src_reg(il, src_reg, 2) if immed is None else il.const(2, immed)
            il.append(to_il_set_reg(il, Registers[dst_reg], il.mult(4, src, to_il_src_reg(il, dst_reg, 2))))
        elif instr == 'mulhi':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.mult(4, il.const(2, immed), to_il_src_reg(il, src_reg, 2))))
        elif instr == 'cmp':
            src = to_il_src_reg(il, src_reg) if immed is None else il.const(4, immed)
            il.append(il.sub(4, to_il_src_reg(il, dst_reg), src, flags='*'))
        elif instr == 'br':
            branch_target = sign_extend(immed, 32) + addr
            il.append(il.jump(il.const(4, branch_target)))
        elif instr in BranchConditionCode:
            branch_target = il.const(4, sign_extend(immed, 32) + addr)
            cond = il.flag_condition(BranchConditionToILCondition[BranchConditionCode.index(instr)])
            cond_branch(il, cond, branch_target)
        elif instr in ['and', 'andi']:
            src = to_il_src_reg(il, dst_reg) if immed is None else il.const(4, immed)
            and_expr = il.and_expr(4, to_il_src_reg(il, src_reg), src, flags='ovsz')
            il.append(to_il_set_reg(il, Registers[dst_reg], and_expr))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr in ['or', 'ori']:
            src = to_il_src_reg(il, dst_reg) if immed is None else il.const(4, immed)
            or_expr = il.or_expr(4, to_il_src_reg(il, src_reg), src, flags='ovsz')
            il.append(to_il_set_reg(il, Registers[dst_reg], or_expr))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr in ['xor', 'xori']:
            src = to_il_src_reg(il, dst_reg) if immed is None else il.const(4, immed)
            xor_expr = il.xor_expr(4, to_il_src_reg(il, src_reg), src, flags='ovsz')
            il.append(to_il_set_reg(il, Registers[dst_reg], xor_expr))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr == 'not':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.not_expr(4, to_il_src_reg(il, src_reg), flags='ovsz')))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr == 'shr':
            shiftee = to_il_src_reg(il, dst_reg)
            shift_amount = il.const(4, immed) if immed is not None else to_il_src_reg(il, src_reg)
            store_in = Registers[dst_reg] if reg3 is None else Registers[reg3]

            shr_expr = il.logical_shift_right(4, shiftee, shift_amount, flags='*')
            il.append(to_il_set_reg(il, store_in, shr_expr))
            # TODO - find a better way of doing this? it falls over if shift amount is 0
            il.append(il.set_flag('cy',
                il.and_expr(1,
                    il.const(1, 1),
                    il.logical_shift_right(4, shiftee, il.sub(4, shift_amount, il.const(1, 1))))))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr == 'shl':
            shiftee = to_il_src_reg(il, dst_reg)
            shift_amount = il.const(4, immed) if immed is not None else to_il_src_reg(il, src_reg)
            store_in = Registers[dst_reg] if reg3 is None else Registers[reg3]

            shl_expr = il.shift_left(4, shiftee, shift_amount, flags='*')
            il.append(to_il_set_reg(il, store_in, shl_expr))
            # TODO - find a better way of doing this? it falls over if shift amount is 0
            il.append(il.set_flag('cy',
                il.and_expr(1,
                    il.const(4, 0x80000000),
                    il.shift_left(4, shiftee, il.sub(4, shift_amount, il.const(1, 1))))))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr == 'sar':
            shiftee = to_il_src_reg(il, dst_reg)
            shift_amount = il.const(4, immed) if immed is not None else to_il_src_reg(il, src_reg)
            store_in = Registers[dst_reg] if reg3 is None else Registers[reg3]

            sar_expr = il.arith_shift_right(4, shiftee, shift_amount, flags='*')
            il.append(to_il_set_reg(il, store_in, sar_expr))
            # TODO - find a better way of doing this? it falls over if shift amount is 0
            il.append(il.set_flag('cy',
                il.and_expr(1,
                    il.const(1, 1),
                    il.logical_shift_right(4, shiftee, il.sub(4, shift_amount, il.const(1, 1))))))
            il.append(il.set_flag('ov', il.const(0, 0)))
        elif instr == 'switch':
            pc_plus_two = il.const(4, il.current_address + 2)
            switch_addr = il.add(4, il.shift_left(4, il.reg(4, Registers[dst_reg]), il.const(1, 1)), pc_plus_two)
            new_pc = il.add(4, il.shift_left(4, il.sign_extend(4, il.load(2, switch_addr)), il.const(1, 1)), pc_plus_two)
            il.append(il.jump(new_pc))
        elif instr == 'zxb':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.zero_extend(4, to_il_src_reg(il, dst_reg, 1))))
        elif instr == 'zxh':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.zero_extend(4, to_il_src_reg(il, dst_reg, 2))))
        elif instr == 'sxb':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.sign_extend(4, to_il_src_reg(il, dst_reg, 1))))
        elif instr == 'sxh':
            il.append(to_il_set_reg(il, Registers[dst_reg], il.sign_extend(4, to_il_src_reg(il, dst_reg, 2))))
        elif instr == 'cmov':
            il.append(il.unimplemented())
        else:
            il.append(il.unimplemented())
        return length

class DefaultCallingConvention(CallingConvention):
    int_arg_regs = ['r6', 'r7', 'r8', 'r9']
    int_return_reg = 'r10'
    # high_int_return_reg = 'r14'

V850.register()
arch = Architecture['V850']
arch.register_calling_convention(DefaultCallingConvention(arch, 'default'))
standalone = arch.standalone_platform
standalone.default_calling_convention = arch.calling_conventions['default']
